"""Self-containment + reachability census for a directory of PE modules.

This is the authority the package builder uses to decide WHICH binaries ship: it computes the
transitive STATIC-import closure of an entry module (llama-server.exe) and reports every other
file in the directory as NOT reachable. Two caveats that keep the live-module check in
build_package.ps1 necessary rather than trusting this alone:

  * a DLL can be pulled in by name at runtime (LoadLibrary), invisible to the import table --
    this is exactly how amdhip64_7.dll drags in amd_comgr.dll and rocm_kpack.dll;
  * a data directory (rocblas/, hipblaslt/, .kpack/) is opened by path, never imported.

So "reachable" here means import-reachable; the builder unions that with the known data dirs and
confirms with a live process module list that nothing extra was needed.

usage:
  python closure.py --dir DIR [--entry NAME]... [--json] [--list]
    --list   print only the reachable ROOT module file names (what to stage), one per line
    --json   machine-readable {reachable, reachable_root, unresolved, unreachable, ...}
    (default) human report
"""
import argparse
import json
import os
import struct


def pe_imports(path):
    """Return the list of DLL names a PE file statically imports.

    Reads the PE/COFF import directory directly (no pefile dependency). The data-directory
    offset is 96 for PE32 and 112 for PE32+; getting that wrong silently reports "no imports"
    for every 64-bit module, which once made a naive parser claim a static CRT.
    """
    with open(path, 'rb') as f:
        d = f.read()
    if d[:2] != b'MZ':
        return []
    pe = struct.unpack_from('<I', d, 0x3C)[0]
    if d[pe:pe + 4] != b'PE\0\0':
        return []
    nsec = struct.unpack_from('<H', d, pe + 6)[0]
    szopt = struct.unpack_from('<H', d, pe + 20)[0]
    opt = pe + 24
    magic = struct.unpack_from('<H', d, opt)[0]
    dd_off = opt + (96 if magic == 0x10b else 112)
    imp_rva, _ = struct.unpack_from('<II', d, dd_off + 8)

    secs = []
    for i in range(nsec):
        o = opt + szopt + i * 40
        vsz, va, rsz, raw = struct.unpack_from('<IIII', d, o + 8)
        secs.append((va, max(vsz, rsz, 1), raw))

    def r2o(rva):
        for va, vs, raw in secs:
            if va <= rva < va + vs:
                return raw + (rva - va)
        return None

    out = []
    o = r2o(imp_rva) if imp_rva else None
    while o:
        name_rva = struct.unpack_from('<I', d, o + 12)[0]
        if not name_rva:
            break
        no = r2o(name_rva)
        if no is None:
            break
        out.append(d[no:d.index(b'\0', no)].decode('ascii', 'replace'))
        o += 20
    return out


def system_dlls():
    system = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32')
    try:
        names = os.listdir(system)
    except OSError:
        return set()
    return set(n.lower() for n in names if n.lower().endswith('.dll'))


def is_always_present(name, sysdlls):
    """True if the import is satisfied by the OS, not by a file we ship.

    api-set / ext-api-set contract DLLs (api-ms-win-crt-*, ext-ms-*) are NOT files on disk --
    the loader resolves them from the in-memory ApiSet schema map -- so they must be treated as
    always available even though a System32 directory listing never contains them. Treating them
    as unresolved is the classic false "package is broken" alarm.
    """
    return name in sysdlls or name.startswith('api-ms-') or name.startswith('ext-ms-')


def analyze(directory, entries):
    by_name, all_rel, dupes = {}, {}, []
    for root, _dirs, names in os.walk(directory):
        for n in names:
            rel = os.path.relpath(os.path.join(root, n), directory)
            all_rel[rel.lower()] = rel
            if n.lower() in by_name:
                dupes.append(rel)
            else:
                by_name[n.lower()] = rel

    sysdlls = system_dlls()
    seen, order, unresolved = set(), [], []
    queue = [e.lower() for e in entries]
    while queue:
        name = queue.pop(0)
        if name in seen:
            continue
        seen.add(name)
        rel = by_name.get(name)
        if rel is None:
            if not is_always_present(name, sysdlls):
                unresolved.append(name)
            continue
        order.append(rel)
        if os.path.splitext(rel)[1].lower() in ('.dll', '.exe'):
            for i in pe_imports(os.path.join(directory, rel)):
                if i.lower() not in seen:
                    queue.append(i.lower())

    reachable = set(r.lower() for r in order)
    return {
        'reachable': sorted(order),
        'reachable_root': sorted(r for r in order if os.path.dirname(r) == ''),
        'unresolved': sorted(set(unresolved)),
        'unreachable': sorted(set(all_rel) - reachable),
        'dupes': sorted(set(dupes)),
        'total_files': len(all_rel),
        'total_bytes': sum(os.path.getsize(os.path.join(directory, r)) for r in all_rel.values()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--entry', action='append', default=None)
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    entries = a.entry or ['llama-server.exe']
    r = analyze(os.path.abspath(a.dir), entries)

    if a.list:
        for x in r['reachable_root']:
            print(x)
        return
    if a.json:
        print(json.dumps(r, indent=2))
        return

    print('=== entry: %s' % ', '.join(entries))
    print('=== reachable root modules (%d) ===' % len(r['reachable_root']))
    for x in r['reachable_root']:
        print('  ' + x)
    print('=== unresolved (%d) ===' % len(r['unresolved']))
    for x in r['unresolved']:
        print('  ' + x)
    if r['dupes']:
        print('=== name collisions (%d) ===' % len(r['dupes']))
        for x in r['dupes']:
            print('  ' + x)
    print('=== unreachable files (%d) ===' % len(r['unreachable']))
    for x in r['unreachable'][:40]:
        print('  ' + x)
    print('total: %d files, %.1f MB' % (r['total_files'], r['total_bytes'] / 1e6))


if __name__ == '__main__':
    main()

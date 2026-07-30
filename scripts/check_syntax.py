"""Syntax checker for all Vasudha Python files."""
import ast
import os
import sys

errors = []
ok_count = 0

for root, dirs, files in os.walk("vasudha"):
    for fname in files:
        if fname.endswith(".py"):
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    source = f.read()
                ast.parse(source, filename=fpath)
                ok_count += 1
            except SyntaxError as e:
                errors.append((fpath, f"SyntaxError: {e}"))
            except UnicodeDecodeError as e:
                errors.append((fpath, f"UnicodeDecodeError: {e}"))

# Also check evaluation/ and tests/
for check_dir in ["evaluation", "tests", "scripts"]:
    if os.path.isdir(check_dir):
        for root, dirs, files in os.walk(check_dir):
            for fname in files:
                if fname.endswith(".py"):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, encoding="utf-8") as f:
                            source = f.read()
                        ast.parse(source, filename=fpath)
                        ok_count += 1
                    except SyntaxError as e:
                        errors.append((fpath, f"SyntaxError: {e}"))
                    except UnicodeDecodeError as e:
                        errors.append((fpath, f"UnicodeDecodeError: {e}"))

print(f"\n{'='*60}")
print(f"Vasudha Syntax Check")
print(f"{'='*60}")
print(f"Files checked: {ok_count + len(errors)}")
print(f"OK:            {ok_count}")
print(f"Errors:        {len(errors)}")

if errors:
    print("\nErrors found:")
    for fpath, msg in errors:
        print(f"  ✗ {fpath}")
        print(f"    {msg}")
    sys.exit(1)
else:
    print("\n✓ All files pass syntax check!")

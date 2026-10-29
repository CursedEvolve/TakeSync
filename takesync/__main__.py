import sys

if len(sys.argv) == 1 or sys.argv[1] == "gui":
    from .gui import main
else:
    from .cli import main

sys.exit(main())

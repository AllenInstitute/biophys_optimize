import argparse, os
import allensdk.core.json_utilities as ju
from pkg_resources import resource_filename
import biophys_optimize

parser = argparse.ArgumentParser(description='hack in paths that strategy will do - passive')
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)
parser.add_argument('--fit_type', default="f6")
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--mu', default=10, type=int)
parser.add_argument('--ngen', default=5, type=int)
parser.add_argument('--sp', default=None, type=str)
args = parser.parse_args()

data = ju.read(args.input)

boph_name = biophys_optimize.__name__

output = {
  "paths": {
    "swc": data["paths"]["swc"],
    "storage_directory": data["paths"]["storage_directory"],
    "preprocess_results": data["paths"]["preprocess_results"],
    "passive_results": os.path.join(data["paths"]["storage_directory"], "consolidated_passive_info.json"),
    "fit_style": resource_filename(boph_name, "fit_styles/%s_fit_style.json" % args.fit_type),
    "compiled_mod_library": resource_filename(boph_name, "x86_64/.libs/libnrnmech.so"),
    "hoc_files": [ "stdgui.hoc", "import3d.hoc", resource_filename(boph_name, "cell.hoc") ]
  },
  "fit_type": args.fit_type,
  "seed": args.seed,
  "mu": args.mu,
  "ngen": args.ngen,
}

if args.sp is not None:
    output["paths"]["starting_population"] = args.sp

ju.write(args.output, output)

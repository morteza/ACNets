from pathlib import Path
import os

try:
  from nipype.interfaces.dcm2nii import Dcm2niix
except ImportError:
  raise "dcm2bids() requires nipype and dcm2niix."


def dcm2bids(dcm_dir: Path,
             out_dir: Path,
             out_filename: str,
             overwrite: bool = True):
  """Converts raw DCM files to BIDS-compatible Nifti and sidecar.

  """

  out_path = out_dir / f'{out_filename}.nii.gz'

  if out_path.exists():
    if not overwrite:
      # warn if out_file exists; dcm2niix adds suffixes if
      # `out_file` exists, makes it incompatible to BIDS spec.
      raise Exception(
          f'[dcm2bids] Nifti already exists: "{out_filename}.nii.gz". '
          f'Delete it or enable overwrite flag in dcm2bids(...) function.'
      )

    # delete existing file if overwritting is allowed
    os.remove(out_path)

  converter = Dcm2niix()

  converter.inputs.source_dir = dcm_dir
  converter.inputs.compress = 'i'
  converter.inputs.output_dir = out_dir
  converter.inputs.out_filename = out_filename

  # DEBUG Path(out_dir).mkdir(parents=True, exist_ok=True)
  # DEBUG converter.inputs.compression = 5
  # DEBUG converter.inputs.has_private = False
  # DEBUG converter.cmdline

  return converter.run()

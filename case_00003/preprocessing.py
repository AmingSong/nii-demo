import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img

# === 路径定义 ===
base = "/Users/songtraming/Desktop/MSU/midi_lab/nii-demo/case_00003/"
img_path = base + "imaging.nii.gz"            # 原始影像
seg_path = base + "segmentation.nii"          # GT
pred_path = base + "case_00003.nii"           # Pred

# ===将原始影像转换为 float32 ===
img = nib.load(img_path)
img_data = img.get_fdata().astype(np.float32)
nib.save(nib.Nifti1Image(img_data, img.affine, img.header), base + "imaging_float32.nii")
print("Saved float32 image")

# ===  重采样掩码到原始影像空间 ===
seg = nib.load(seg_path)
pred = nib.load(pred_path)

seg_res = resample_to_img(seg, img, interpolation="nearest")
pred_res = resample_to_img(pred, img, interpolation="nearest")

nib.save(seg_res, base + "segmentation_aligned.nii")
nib.save(pred_res, base + "case_00003_aligned.nii")
print("Resampled GT and Pred to image space")

# ===  验证对齐 ===
print("Shape:", img.shape, seg_res.shape, pred_res.shape)
print("Affine match:",
      np.allclose(img.affine, seg_res.affine),
      np.allclose(img.affine, pred_res.affine))


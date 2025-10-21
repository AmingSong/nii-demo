import sys, numpy as np, nibabel as nib

def to_3d_int_label(img):
    data = img.get_fdata()  # float64 by default
    aff  = img.affine
    hdr  = img.header

    if data.ndim == 4:
        # Move channel axis to last if it's first: (C,X,Y,Z) -> (X,Y,Z,C)
        if data.shape[0] in (3,4) and data.shape[-1] not in (3,4):
            data = np.moveaxis(data, 0, -1)
        # Argmax over channel to get {0,1,2,...}; here we assume 3 classes (kidney,tumor,cyst)
        label = np.argmax(data, axis=-1) + 1  # -> {1,2,3}
        # Optional background: if all channels ~0, mark 0
        bg = (np.max(data, axis=-1) <= 0.0)
        label[bg] = 0
    elif data.ndim == 3:
        # Round floats to nearest int; keep only {0,1,2,3}
        label = np.rint(data).astype(np.int16)
        valid = np.isin(label, [0,1,2,3])
        label[~valid] = 0
    else:
        raise RuntimeError(f"Unsupported ndim={data.ndim}, expected 3D or 4D")

    out = nib.Nifti1Image(label.astype(np.int16), aff, hdr)
    out.set_data_dtype(np.int16)
    return out, label.shape, label.dtype

def main():
    if len(sys.argv) != 3:
        print("Usage: python make_labels_int.py <src.nii> <dst.nii>")
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    img = nib.load(src)
    out, shape, dtype = to_3d_int_label(img)
    nib.save(out, dst)
    print(f"saved: {dst}, shape: {shape}, dtype: {dtype}")

if __name__ == "__main__":
    main()


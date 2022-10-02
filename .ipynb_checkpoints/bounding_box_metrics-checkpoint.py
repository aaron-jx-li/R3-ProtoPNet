import numpy as np


# Calculates the Overlap of Prototypes in the Same Class over shared images
'''
bb_holder columns:
0: image index in the entire dataset
1: height start index
2: height end index
3: width start index
4: width end index
5: (optional) class identity
'''
def bounding_box_overlap(bb_holder, n_prototypes, n_classes):
    protos_per_class = int(n_prototypes / n_classes)
    
    global_overlap = 0
    for i in range(0, n_classes):
        start_idx = i*protos_per_class
        end_idx = (i+1)*protos_per_class
        
        bb_class_specific = bb_holder[start_idx : end_idx]
        
        # First Column is the Img ID
        u, c = np.unique(bb_class_specific[:, 0], return_counts=True)
        shared_imgs = u[c > 1]
        
        bb_shared = [x for x in bb_class_specific if x[0] in shared_imgs]
        
        # Add Normalized Class Overlap (/ by # of overlaps computed)
        if len(bb_shared) > 0:
            class_overlap = compute_overlap(bb_shared) / len(bb_shared)
            global_overlap += class_overlap
        
    # Return the normalized overlap
    return global_overlap / n_classes
        
# Compute overlap within one class
def compute_overlap(bb_object):
    total_overlap = 0
    for j in range(0, len(bb_object)):
        for k in range(0, len(bb_object)):
            if j == k:
                continue

            l = bb_object[j]
            r = bb_object[k]

            dy = min(l[2], r[2]) - max(l[1], r[1])
            dx = min(l[4], r[4]) - max(l[3], r[3])
            
            if dx <= 0 or dy <= 0:
                overlap = 0
            else:
                SI = dx*dy
                SL = (l[2] - l[1])*(l[4] - l[3])
                SR = (r[2] - r[1])*(r[4] - r[3])
                SU = SL + SR - SI

                # Calculate the ratio to get overlap
                # Taking the min: get percent of smallest
                # bounding box is intersecting with the other bb
                overlap = SI / min(SL, SR)

            total_overlap += overlap
            
    # Remove Duplicates
    total_overlap /= 2
            
    return total_overlap


"""
# Test Values to Ensure that Overlap Computed Correctly
tm = np.array([
    [1, 0, 10, 0, 10, 5],
    [1, 0, 10, 0, 10, 5],
])

tm2 = np.array([
    [1, 10, 20, 10, 20, 5],
    [1, 0, 10, 0, 10, 5],
])

tm3 = np.array([
    [1, 5, 15, 5, 15, 5],
    [1, 0, 10, 0, 10, 5],
])

tm4 = np.array([
    [1, 5, 15, 0, 10, 5],
    [1, 0, 10, 0, 10, 5],
])

tm5 = np.array([
    [1, 0, 10, 0, 10, 5],
    [1, 2, 8, 2, 8, 5],
])


"""



   

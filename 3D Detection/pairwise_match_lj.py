import sys
from collections import defaultdict
import numpy as np
import os

import h5py
import fast64counter

import time
# from util import Util
import matplotlib.pyplot as plt
from tifffile import imsave


Debug = False

# block1_path, block2_path, direction, halo_size, outblock1_path, outblock2_path = sys.argv[1:]
def pair_match(block1,block2, direction, halo_size):
    # block1_path = './data/seg1.h5'
    # block2_path = './data/seg2_unique.h5'
    # outblock1_path = 'out1.h5'
    # outblock2_path = 'out2.h5'
    # direction = 1
    # halo_size = 10
    # plt.imshow(block1[4, :, :])
    # plt.figure()
    # plt.imshow(block2[0, :, :])
    direction = int(direction)
    halo_size = int(halo_size)

###############################
# Note: direction indicates the relative position of the blocks (1, 2, 3 =>
# adjacent in X, Y, Z).  Block1 is always closer to the 0,0,0 corner of the
# volume.
###############################

###############################
# Note that we are still in matlab hdf5 coordinates, so everything is stored ZYX
###############################

###############################
#Change joining thresholds here
###############################
#Join 1 (less joining)
    # auto_join_pixels = 20000 # Join anything above this many pixels overlap
    # minoverlap_pixels = 2000 # Consider joining all pairs over this many pixels overlap
    # minoverlap_dual_ratio = 0.7 # If both overlaps are above this then join
    # minoverlap_single_ratio = 0.9# If either overlap is above this then join

# Join 2 (more joining)
    auto_join_pixels = 1000 # Join anything above this many pixels overlap
    minoverlap_pixels = 300 # Consider joining all pairs over this many pixels overlap
    minoverlap_dual_ratio = 0.5 # If both overlaps are above this then join
    minoverlap_single_ratio = 0.8 # If either overlap is above this then join


    # print 'Running pairwise matching', " ".join(sys.argv[1:])



    # assert block1.size == block2.size

    # append the blocks, and pack them so we can use the fast 64-bit counter
    # stacked = np.vstack((block1, block2))
    # inverse, packed = np.unique(stacked, return_inverse=True)
    # packed = packed.reshape(stacked.shape)
    # packed_block1 = packed[:block1.shape[0], :, :]
    # packed_block2 = packed[block1.shape[0]:, :, :]

    # Adjust for Matlab HDF5 storage order
    # direction = 3 - direction
    direction = direction - 1

    stacked = np.concatenate((block1, block2), axis=direction)
    inverse, packed = np.unique(stacked, return_inverse=True)
    packed = packed.reshape(stacked.shape)
    if direction == 0:
        packed_block1 = packed[:block1.shape[0], :, :]
        packed_block2 = packed[block1.shape[0]:, :, :]
    elif direction == 1:
        packed_block1 = packed[:, :block1.shape[1], :]
        packed_block2 = packed[:, block1.shape[1]:, :]
    else:
        packed_block1 = packed[:, :, :block1.shape[2]]
        packed_block2 = packed[:, :, block1.shape[2]:]


    # extract overlap

    lo_block1 = [0, 0, 0]
    hi_block1 = [None, None, None]
    lo_block2 = [0, 0, 0]
    hi_block2 = [None, None, None]



    # Adjust overlapping region boundaries for direction
    lo_block1[direction] = - 1 * halo_size
    hi_block2[direction] = 1 * halo_size


    block1_slice = tuple(slice(l, h) for l, h in zip(lo_block1, hi_block1))
    block2_slice = tuple(slice(l, h) for l, h in zip(lo_block2, hi_block2))
    packed_overlap1 = packed_block1[block1_slice]
    packed_overlap2 = packed_block2[block2_slice]
    print("block1", block1_slice, packed_overlap1.shape)
    print("block2", block2_slice, packed_overlap2.shape)

    counter = fast64counter.ValueCountInt64()
    counter.add_values_pair32(packed_overlap1.astype(np.int32).ravel(), packed_overlap2.astype(np.int32).ravel())
    overlap_labels1, overlap_labels2, overlap_areas = counter.get_counts_pair32()

    areacounter = fast64counter.ValueCountInt64()
    areacounter.add_values(packed_overlap1.ravel())
    areacounter.add_values(packed_overlap2.ravel())
    areas = dict(zip(*areacounter.get_counts()))

    to_merge = []
    to_steal = []
    merge_dict = {}
    for l1, l2, overlap_area in zip(overlap_labels1, overlap_labels2, overlap_areas):
        # if inverse[l2] == 8828 and inverse[l1] == 2193:
        #     bug = 2
        if l1 == 0 or l2 == 0:
            continue
        if ((overlap_area > auto_join_pixels) or
            ((overlap_area > minoverlap_pixels) and
             ((overlap_area > minoverlap_single_ratio * areas[l1]) or
              (overlap_area > minoverlap_single_ratio * areas[l2]) or
              ((overlap_area > minoverlap_dual_ratio * areas[l1]) and
               (overlap_area > minoverlap_dual_ratio * areas[l2]))))):
            if inverse[l2] in merge_dict:
                if overlap_area < merge_dict[inverse[l2]][1]:
                    continue

            if inverse[l1] != inverse[l2]:
                # print "Merging segments {0} and {1}.".format(inverse[l1], inverse[l2])
                to_merge.append((inverse[l1], inverse[l2]))
                merge_dict[inverse[l2]]=(inverse[l1],overlap_area)
        else:
            # print "Stealing segments {0} and {1}.".format(inverse[l1], inverse[l2])
            to_steal.append((overlap_area, l1, l2))

    # handle merges by rewriting the inverse
    merge_map = dict(reversed(sorted(s)) for s in to_merge)
    ########lj add###############
    for idx, val in enumerate(inverse):
        if val in merge_map:
            while val in merge_map:
                val = merge_map[val]
            inverse[idx] = val
    # Remap and merge
    # out1 = h5py.File(outblock1_path + '_partial', 'w')
    # out2 = h5py.File(outblock2_path + '_partial', 'w')
    # outblock1 = out1.create_dataset('/labels', block1.shape, block1.dtype, chunks=label_chunks, compression='gzip')
    # outblock2 = out2.create_dataset('/labels', block2.shape, block2.dtype, chunks=label_chunks, compression='gzip')
    # outblock1[...] = inverse[packed_block1]
    # outblock2[...] = inverse[packed_block2]

    # Util.view(outblock1[40,:,:],large=True)
    # Util.view(outblock2[0,:,:],large=True)
    # out_one = h5py('out12.h5','w')
    # plt.figure()
    # plt.imshow(block2[0,:,:]==8836)
    outblock1 = inverse[packed_block1]
    outblock2 = inverse[packed_block2]

    # Util.view(outblock1[2,:,:],large=True,file='temp3.png')
    # Util.view(outblock2[0,:,:],large=True,file='temp4.png')
    # plt.figure()
    # plt.imshow(outblock2[0,:,:])
    # out_one = np.vstack((outblock1, outblock2[halo_size:]))
    # out_one = np.vstack((outblock1[:outblock1.shape[0]-1], outblock2))
    lo_block1 = [0, 0, 0]
    hi_block1 = [None, None, None]
    lo_block2 = [0, 0, 0]
    hi_block2 = [None, None, None]

############ obtain overlap region ############
    lo_block1[direction] = - 1 * halo_size
    hi_block2[direction] = 1 * halo_size

    block1_slice = tuple(slice(l, h) for l, h in zip(lo_block1, hi_block1))
    block2_slice = tuple(slice(l, h) for l, h in zip(lo_block2, hi_block2))
    block_overlap1 = outblock1[block1_slice]
    block_overlap2 = outblock2[block2_slice]

    overlap = np.where(block_overlap1>0, block_overlap1, block_overlap2)
###################################
    lo_block1 = [0, 0, 0]
    hi_block1 = [None, None, None]
    lo_block2 = [0, 0, 0]
    hi_block2 = [None, None, None]

    lo_block2[direction] = 1 * halo_size
    block22_slice = tuple(slice(l, h) for l, h in zip(lo_block2, hi_block2))

    hi_block1[direction] = 1 * (block1.shape[direction]-halo_size)
    block1_slice = tuple(slice(l, h) for l, h in zip(lo_block1, hi_block1))

    out_one = np.concatenate((outblock1[block1_slice], overlap, outblock2[block22_slice]), axis=direction)
    # return outblock1, outblock2
    return out_one


if __name__ == '__main__':
    out = np.load('out.npy')
    mask = np.load('mask.npy')
    out = pair_match(out, mask, direction=1, halo_size=28)
    imsave('out.tif', out)

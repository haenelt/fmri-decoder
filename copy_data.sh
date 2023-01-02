#!/bin/bash
# This is small script that copies all necessary input data of my ocular dominance 
# columns project and stores it into a common output directory. 
# 
# copy_data <subj> <sess> <dir_out>
# 
# Args:
#   subj    - subject name.
#   sess    - session name.
#   dir_out - Output directory.
#

# base directory on my workstation
DIR_BASE="/data/pt_01880/Experiment1_ODC"

# input arguments
subj=$1
sess=$2
dir_out=$3

dir_surf=$dir_out/ana/surf
dir_deform=$dir_out/ana/deform
dir_label=$dir_out/ana/label
dir_func=$dir_out/func
dir_loc=$dir_out/loc

mkdir -p $dir_surf
mkdir -p $dir_deform
mkdir -p $dir_label
mkdir -p $dir_func
mkdir -p $dir_loc

# functional data
for i in {1..10}
do
    # time series data
    file_data=$DIR_BASE/$subj/odc/$sess/Run_$i/udata.nii
    cp $file_data $dir_func/run_${i}_bold.nii

    # condition logfiles
    file_cond=$DIR_BASE/$subj/odc/$sess/Run_$i/logfiles/${subj}_${sess}_Run${i}_odc_Cond.mat
    cp $file_cond $dir_func/run_${i}_events.mat
done

# surfaces
for i in {0..10}
do
    file_layer_left=$DIR_BASE/$subj/anatomy/layer/lh.layer_$i
    file_layer_right=$DIR_BASE/$subj/anatomy/layer/rh.layer_$i
    cp $file_layer_left $dir_surf/lh.layer_$i
    cp $file_layer_right $dir_surf/rh.layer_$i
done

# deformation
file_deform=$DIR_BASE/$subj/deformation/odc/$sess/source2target.nii.gz
cp $file_deform $dir_deform/transformation.nii.gz

# localizer
file_loc1_left=$DIR_BASE/$subj/retinotopy/avg/sampled/ecc_snr_avg/lh.ecc_snr_avg_layer_5.mgh
file_loc1_right=$DIR_BASE/$subj/retinotopy/avg/sampled/ecc_snr_avg/rh.ecc_snr_avg_layer_5.mgh
file_loc2_left=$DIR_BASE/$subj/retinotopy/avg/sampled/pol_snr_avg/lh.pol_snr_avg_layer_5.mgh
file_loc2_right=$DIR_BASE/$subj/retinotopy/avg/sampled/pol_snr_avg/rh.pol_snr_avg_layer_5.mgh
cp $file_loc1_left $dir_loc/lh.localizer1.mgh
cp $file_loc1_right $dir_loc/rh.localizer1.mgh
cp $file_loc2_left $dir_loc/lh.localizer2.mgh
cp $file_loc2_right $dir_loc/rh.localizer2.mgh

# label (v1)
file_v1_left=$DIR_BASE/$subj/anatomy/label/lh.v1.label
file_v1_right=$DIR_BASE/$subj/anatomy/label/rh.v1.label
cp $file_v1_left $dir_label/lh.v1.label
cp $file_v1_right $dir_label/rh.v1.label

# label (fov)
file_fov_left=$DIR_BASE/$subj/anatomy/label/lh.fov.label
file_fov_right=$DIR_BASE/$subj/anatomy/label/rh.fov.label
cp $file_fov_left $dir_label/lh.fov.label
cp $file_fov_right $dir_label/rh.fov.label

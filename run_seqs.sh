#!/bin/bash

export INPATH=/media/prakhar/BIG_BAG/Capstone/Loris_RGBD

export s1=2
./Examples/RGB-D/rgbd_tum_seq ./Vocabulary/ORBvoc.txt ./Examples/RGB-D/LORIS.yaml ${INPATH}/office1-${s1}/ ./Examples/RGB-D/associations/office1-${s1}.txt dataset-office1-${s1}_RGBD

export s1=3
./Examples/RGB-D/rgbd_tum_seq ./Vocabulary/ORBvoc.txt ./Examples/RGB-D/LORIS.yaml ${INPATH}/office1-${s1}/ ./Examples/RGB-D/associations/office1-${s1}.txt dataset-office1-${s1}_RGBD

export s1=4
./Examples/RGB-D/rgbd_tum_seq ./Vocabulary/ORBvoc.txt ./Examples/RGB-D/LORIS.yaml ${INPATH}/office1-${s1}/ ./Examples/RGB-D/associations/office1-${s1}.txt dataset-office1-${s1}_RGBD

export s1=5
./Examples/RGB-D/rgbd_tum_seq ./Vocabulary/ORBvoc.txt ./Examples/RGB-D/LORIS.yaml ${INPATH}/office1-${s1}/ ./Examples/RGB-D/associations/office1-${s1}.txt dataset-office1-${s1}_RGBD

export s1=6
./Examples/RGB-D/rgbd_tum_seq ./Vocabulary/ORBvoc.txt ./Examples/RGB-D/LORIS.yaml ${INPATH}/office1-${s1}/ ./Examples/RGB-D/associations/office1-${s1}.txt dataset-office1-${s1}_RGBD

export s1=7
./Examples/RGB-D/rgbd_tum_seq ./Vocabulary/ORBvoc.txt ./Examples/RGB-D/LORIS.yaml ${INPATH}/office1-${s1}/ ./Examples/RGB-D/associations/office1-${s1}.txt dataset-office1-${s1}_RGBD

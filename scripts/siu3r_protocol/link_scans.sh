# link fully-extracted SIU3R val scans into the SIU3R-layout root (idempotent)
S=/mnt/workspace/Jensen/project/3DGS/data/siu3r_val_full; R=/mnt/workspace/Jensen/project/3DGS/data/siu3r_protocol_root/val
for d in $S/scene*/.extracted; do s=$(basename $(dirname $d)); [ -e $R/$s ] || ln -s $S/$s $R/$s; done
ls $R | wc -l

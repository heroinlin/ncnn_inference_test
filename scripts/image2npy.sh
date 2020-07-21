cd ../
image_path="./samples/data.png"
save_path="./samples/data.npy"
width=224
height=224
mean='0.4914 0.4822 0.4465'
stddev='0.247 0.243 0.261'
divisor=255.0
color_format=RGB
python scripts/image2npy.py -i $image_path -s $save_path --width $width  --height $height  --mean $mean --stddev $stddev --divisor $divisor --color_format $color_format
from PIL import Image
import sys

im = Image.open(sys.argv[1])
rgb_im = im.convert('RGB')

#w, h
width, height = im.size

# #save Image
# output_image = Image.new("RGB", (width, height), "white")

# #turn around the image
# for w in range(0, width):
# 	for h in range(0, height):
# 		#r, g, b = rgb_im.getpixel((w, h))
# 		rgb = im.getpixel((w, h))
# 		output_image.putpixel((width-w-1, height-h-1), rgb)
# output_image.save("ans2.png")

out = im.transpose(Image.ROTATE_180)
out.save("ans2.png")

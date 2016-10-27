def convert(imgf, labelf, outf, outfLabels, n):
    f = open(imgf, "rb")
    o1 = open(outf, "w")
    l = open(labelf, "rb")
    o2 = open(outfLabels, "w")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o2.write(str(image[0])+"\n")#write label separately
        #take pixels without label
        o1.write(",".join(str(pix) for pix in image[1:])+"\n")
    f.close()
    o1.close()
    o2.close()
    l.close()

convert("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
        "train.csv", "label_train.csv", 60000)
convert("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte",
        "test.csv", "label_test.csv", 10000)

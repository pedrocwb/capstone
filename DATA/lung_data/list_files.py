import os



def absoluteFilePaths(foo="training"):


    gt  = '{}/gt_masks/'.format(foo)
    img = '{}/lung_images/'.format(foo)

    for (im, mask) in zip(os.listdir(img), os.listdir(gt)):
            print(os.path.abspath(img + im) + " " + os.path.abspath(gt + mask))


absoluteFilePaths()
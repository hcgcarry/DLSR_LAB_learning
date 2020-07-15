import matplotlib.pyplot as plt
def showsubplot(images, imagesTitle=None):
    row = int(len(images) / 6) + 1
    # list轉換成np.array

    if imagesTitle is None:
        imagesTitle = list(range(len(images)))

    plt.figure(figsize=(30, 30))

    for index in range(len(images)):
        plt.subplot(row, 6, index + 1)
        plt.imshow(images[index], cmap="gray")

        plt.title(imagesTitle[index])
        # ticks
    plt.show();

from placer import Placer
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset_path = [f"../objects/{i}.jpg" for i in range(1, 11)]
    background_path = "../objects/0.jpg"
    placer = Placer(background_path, dataset_path)
    placer.load_image("../tests/1.jpg")
    print(placer.place_objects(1e-3, 1000))
    plt.imshow(placer.draw_objects())
    plt.show()

import tensorflow_datasets as tfds

print("Building builder...")
builder = tfds.builder("coco/2017")

print("Starting download_and_prepare")
builder.download_and_prepare()

print("done")
print(builder.info)
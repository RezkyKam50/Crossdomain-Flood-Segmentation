
# dummy dataset 4 testing
git xet install
git clone https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense


# real dataset Sen1Floods11
gcloud init
gsutil -m rsync -r gs://sen1floods11 ./datasets/


# real dataset BuildDMG
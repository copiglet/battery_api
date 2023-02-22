
#
#curl http://127.0.0.1:8081/ai/v1/ng_detect/ng002 -v -o ./ngout-demo.json

vid="S6046CCBPB10198218"
cp -R /mnt/nas03/phenomx/huvio/new_test_data/plus/plus/omission/${vid} ../inference/images/

ls -l ../inference/images

curl http://127.0.0.1:8081/ai/v1/ng_detect/${vid} -v -o ./ngout-${vid}-demo.json

cat ./ngout-${vid}-demo.json




#
# curl -X POST http://127.0.0.1:8081/ai/v1/ng_batch_detect -d "rootdir=/mnt/nas03/phenomx/huvio/new_test_data/plus" -v -o ./batch-test-out.txt


curl -X POST http://127.0.0.1:8081/ai/v1/ng_batch_detect -d "rootdir=/mnt/nas03/phenomx/huvio/huvio_docker/input" -v -o ./batch-test-out.txt


# curl -X POST http://127.0.0.1:8081/ai/v1/ng_batch_detect -d "rootdir=/mnt/nas03/phenomx/huvio/new_test_data" -v -o ./batch-test-out.txt
echo "------------"
sleep 1

cat ./batch-test-out.txt



#
# path param : /a_sae_img/a_debug_print/a_image_path
# form param : image_path=.....

# set debug all and change image_path
# curl -X POST http://127.0.0.1:8081/ai/v1/ng_config/1/1/1 -d "image_path=/tmp/test" -v -o ./config-out.txt

# set debug disable and change image path
curl -X POST http://127.0.0.1:8081/ai/v1/ng_config/0/0/1 -d "image_path=./inference/images" -v -o ./config-out.txt

cat ./config-out.txt


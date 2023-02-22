# test hvdetect.py
vid="S6046CCBPB10198218"

# ready data
cp -R ../new_test_data/plus/plus/omission/${vid} ./inference/images/

# module test
# DEBUG_PRINT=true python hvdetect.py ${vid}


#      curl http://127.0.0.1:8081/ai/v1/ng_detect/S6046CCBPB10198218 -v


echo "curl http://127.0.0.1:8081/ai/v1/ng_detect/${vid} -v"
echo ""

# web test
# DEBUG_PRINT=true python hvweb.py > web.log 2>&1 &
DEBUG_PRINT=true python hvweb.py


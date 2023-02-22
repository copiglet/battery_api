'''
name : hvweb.py
desc : huvio web inferface
'''
import argparse
import time
from pathlib import Path
import os, sys
import shutil
import json
from bottle import request, response, template, abort, run, redirect
from bottle import static_file
from bottle import Bottle

from hvdetect import HvDetect


save_img = os.environ.get('SAVE_IMG', "false").lower() == "true"
debug_print = os.environ.get('DEBUG_PRINT', "true").lower() == "true"
image_path = os.environ.get('IMAGE_PATH', './inference/images')
web_debug = os.environ.get('WEB_DEBUG', "false").lower() == "true"

print('==========MAIN-CONFIG===========')
print(f" - save_img:{save_img}")
print(f" - debug_print:{debug_print}")
print(f" - image_path:{image_path}")
print(f" - web_debug:{web_debug}")
print("")

# get detector
detector = HvDetect(save_img, debug_print, image_path, web_debug)
        
# bottle app
app = Bottle()

# shutdown web
@app.get('/ai/v1/shutdown/<a_code>')
def ng_shutdown(a_code):
    print(f"shutdown request by user: code: {a_code}")
    if a_code == "999":
        print("--------------------BYE--------------------")
        sys.exit("shutdown by user")
    else:
        print("shutdown code not matched !")
        return "PASS"

# set config
@app.post('/ai/v1/ng_config/<a_save_img>/<a_debug_print>/<a_image_path>/<a_web_debug>')
def ng_config(a_save_img, a_debug_print, a_image_path, a_web_debug):

    save_img = a_save_img == "1"
    debug_print = a_debug_print == "1"
    web_debug = a_web_debug == "1"
    if a_image_path == "1":
        image_path = request.params.get('image_path')

    print("=========SET CONFIG========")
    print(f" - save_img:{save_img}")
    print(f" - debug_print:{debug_print}")
    print(f" - image_path:{image_path}")
    print(f" - web_debug:{web_debug}")

    detector.setconfig(save_img, debug_print, image_path, web_debug)

    return f"SET:{save_img}, {debug_print}, {image_path}"

# run prediction by cell_id 
@app.get('/ai/v1/ng_detect/<req_id>')
def ng_detect(req_id):
    code, res_js = detector.run_detect(req_id)
    if debug_print:
        print("----")
        print(code)
        print(json.dumps(res_js, indent=4))
        print("----")

    if code == 200:
        return res_js
    elif code == 400:
        abort(400, res_js)
    else:
        abort(404, res_js)

# run prediction by file 
@app.get('/ai/v1/ng_detect/file/<req_id>/<fn>')
def ng_detect_file(req_id, fn):
    print(f"detect_file : {req_id}, {fn}")
    code, res_js = detector.run_detect(req_id, '', fn)
    if debug_print:
        print("----")
        print(code)
        print(json.dumps(res_js, indent=4))
        print("----")

    if code == 200:
        return res_js
    elif code == 400:
        abort(400, res_js)
    else:
        abort(404, res_js)







def get_png_sub_dirs(rootdir, out_list):
    for file1 in os.listdir(rootdir):
        d = os.path.join(rootdir, file1)
        if os.path.isdir(d):
            pngcnt = 0
            for ff in os.listdir(d):
                f = os.path.join(d, ff)
                if f.endswith(".png"):
                    pngcnt += 1
            if pngcnt > 9:
                out_list.append((rootdir, file1, d))
            get_png_sub_dirs(d, out_list)

# ================================================
PATH_WEBD="./static/webd"
def cleanWebDebug():
    if not os.path.exists(PATH_WEBD):
        os.makedir(PATH_WEBD)
    os.system("rm -f " + PATH_WEBD + "/*")

def writeWebDebug(fpath, jso):
    with open(os.path.join(PATH_WEBD, fpath), "w") as outfile:
        json.dump(jso, outfile)
# ================================================



# run batch prediction
@app.post('/ai/v1/ng_batch_detect')
def ng_batch_detect():
    rootdir = request.params.get('rootdir')
    dstdir = os.path.join(rootdir, 'output')
    fn_res = os.path.join(dstdir, "result.csv") 

    if not os.path.exists(dstdir):
        os.mkdir(dstdir)

    out_list = [] # clear
    get_png_sub_dirs(rootdir, out_list)
    print('batch enter:', rootdir, " total:", len(out_list))
    print("====")

    # init and set
    pred_list = []
    for i, item in enumerate(out_list):
        pred_list.append({"no": i, "item": item})
    if web_debug:
        cleanWebDebug()
        writeWebDebug('pred_list.json', pred_list)

    dorun = True
    out_lines = []
    for jobitem in pred_list:
        item = jobitem["item"]
        pt_path, req_id, f_dir = item

        if dorun:
            code, res_js = detector.run_detect(req_id, pt_path)
            if web_debug:
                writeWebDebug(f'{jobitem["no"]:05}_item.json', {"no": jobitem["no"], "result":
                    res_js})

            txtcode = ["", "ERROR", "GOOD"] 
            if code != 200:
                print('batch run:', f_dir, ':ERROR')
            else:
                if debug_print:
                    print('batch run:', f_dir, ':', res_js["ng"])
                out_line = [pt_path, req_id, txtcode[1+res_js["ng"]]]
                sums = [0, 0, 0]
                for dd in res_js["detail"]:
                    sums[1+dd["ng"]] += 1
                out_line.append(str(sums[0]))
                out_line.append(str(sums[1]))
                out_line.append(str(sums[2]))

                for dd in res_js["detail"]:
                    out_line.append(txtcode[1+dd["ng"]])
                out_lines.append(out_line)
                with open(fn_res, "a") as fileobj:
                    fileobj.write(",".join(out_line) + "\n")

    return {"dstdir": dstdir, "total": len(out_list), "result_file": fn_res}


#
# run :   DEBUG_PRINT=false python detect_web.py 
#
if __name__ == '__main__':
    print('API SERVER RUN')
    port_num = int(os.environ.get('WEB_PORT', "8081"))
    run(app, host="0.0.0.0", port=port_num, debug=True, reloader=False)


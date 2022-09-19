import get_android
# get_android.setTflite("movenet_float32.tflite")
#拉流
# get_android.androidOb("ip_camera_config.json",1,960,540,4)
#原生
# get_android.androidOb("native_config.json",4,1920,1080,1)
#usb
get_android.androidOb("uvc_config.json",6,1920,1080,1)

get_android.run(get_android.ai_worker2)


#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <glib.h>

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);

    // Create RTSP server
    GstRTSPServer *server = gst_rtsp_server_new();
    GstRTSPMountPoints *mounts = gst_rtsp_server_get_mount_points(server);

    // Create RTSP media factory
    GstRTSPMediaFactory *factory = gst_rtsp_media_factory_new();

    // Define the GStreamer pipeline for streaming
    gst_rtsp_media_factory_set_launch(factory,
        "( nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM),width=1920,height=1080 ! "
        "nvvidconv ! video/x-raw,format=I420 ! "
        "x264enc tune=zerolatency bitrate=5000 speed-preset=ultrafast ! "
        "rtph264pay name=pay0 pt=96 )");

    // Allow multiple clients to share the stream
    gst_rtsp_media_factory_set_shared(factory, TRUE);

    // Mount the factory at the specified path
    gst_rtsp_mount_points_add_factory(mounts, "/stream", factory);
    g_object_unref(mounts);

    // Start the RTSP server
    gst_rtsp_server_attach(server, NULL);
    g_print("RTSP server is running at rtsp://127.0.0.1:8554/stream\n");

    // Run the main loop
    GMainLoop *loop = g_main_loop_new(NULL, FALSE);
    g_main_loop_run(loop);

    return 0;
}


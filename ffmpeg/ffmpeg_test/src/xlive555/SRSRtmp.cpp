#include "xlive555/SRSRtmp.h"

#include "srs_librtmp.h"

#include "console_log.h"

SrsRtmp::SrsRtmp()
    : hdl_()
    , url_()
{ 
}

SrsRtmp::SrsRtmp(std::string url) 
    : hdl_()
    , url_(std::move(url))
{
};

SrsRtmp::~SrsRtmp() 
{
    if (hdl_)
        Destroy(hdl_);
};

int SrsRtmp::Init()
{
    if (url_.empty())
        return -1;
    if (hdl_)
        return -1;
    int ret = Create(url_.c_str(), &hdl_);
    return ret;
}

int SrsRtmp::SendH264(void* frameData, int frameLen, uint64_t pts)
{
    int ret = 0;
    if (!hdl_)
        ret -1;
    static int total_num = 0;

#if 0
    //为降低CPU，不采用
            // @remark, to decode the file.
    char* p = (char*)frameData;
    int count = 0;
    for (; p < frameData + frameLen;) {
        // @remark, read a frame from file buffer.
        char* data = NULL;
        int size = 0;
        int nb_start_code = 0;
        if (read_h264_frameEx((char*)frameData, (int)frameLen, &p, &nb_start_code, &data, &size) < 0) {
            srs_human_trace("read a frame from file buffer failed.");
            goto rtmp_destroy;
        }

        // send out the h264 packet over RTMP
        unsigned char* pV = (unsigned char*)data;
        dts += 40;
        pts += 40;
        printf("%02x %02x %02x %02x \n", pV[0], pV[1], pV[2], pV[3]);
        ret = srs_h264_write_raw_frames(m_rtmp, data, size, pts, dts);
    }
#else
    ret = srs_h264_write_raw_frames(hdl_, (char*)frameData, frameLen, pts, pts);
#endif
    if (ret != 0) {
        logPrintWarn("PVS VSTREAM srs rtmp, send h264 raw data ret=%d, rtmp:'%s'\n", ret, url_.c_str());
    }

    if (ret != 0 && 0)
    {
        if (srs_h264_is_dvbsp_error(ret))
        {
            if (total_num % 50 == 0)
            {
                logPrintInfo("PVS VSTREAM srs rtmp, ignore drop video error, code=%d ,rtmp:'%s'\n", ret, url_.c_str());
            }
            total_num++;
        }
        else if (srs_h264_is_duplicated_sps_error(ret))
        {
            //printf("RtmpClientThread::rtmpClientThread_rtmpSendH264Frame: ignore duplicated sps, code=%d\n", ret);
        }
        else if (srs_h264_is_duplicated_pps_error(ret))
        {
            //printf("RtmpClientThread::rtmpClientThread_rtmpSendH264Frame: ignore duplicated pps, code=%d\n", ret);
        }
        else
        {
            logPrintError("PVS VSTREAM srs rtmp, send h264 raw data failed. ret=%d, rtmp:'%s'\n", ret, url_.c_str());
        }

#if 0
        // 5bits, 7.3.1 NAL unit syntax,
        // H.264-AVC-ISO_IEC_14496-10.pdf, page 44.
        //  7: SPS, 8: PPS, 5: I Frame, 1: P Frame, 9: AUD, 6: SEI
        u_int8_t nut = (char)data[nb_start_code] & 0x1f;
        srs_human_trace("sent packet: type=%s, time=%d, size=%d, fps=%.2f, b[%d]=%#x(%s)",
            srs_human_flv_tag_type2string(SRS_RTMP_TYPE_VIDEO), dts, size, fps, nb_start_code, (char)data[nb_start_code],
            (nut == 7 ? "SPS" : (nut == 8 ? "PPS" : (nut == 5 ? "I" : (nut == 1 ? "P" : (nut == 9 ? "AUD" : (nut == 6 ? "SEI" : "Unknown")))))));
#endif
    }
    return ret;
}

int SrsRtmp::Reset(std::string url)
{
    if (hdl_) {
        hdl_ = nullptr;
        Destroy(hdl_);
    }
    url_ = std::move(url);
    return 0;
}

int SrsRtmp::Reset()
{
    if (hdl_) {
        hdl_ = nullptr;
        Destroy(hdl_);
    }
    return 0;
}

const std::string& SrsRtmp::URL() const
{
    return url_;
}

int SrsRtmp::Create(const char* url, srs_rtmp_t* p_rtmp_hdl)
{
    srs_rtmp_t hdl = srs_rtmp_create(url);
    if (!hdl) {
        logPrintWarn("PVS VSTREAM srs rtmp, create failed, rtmp:'%s'\n", url);
        return -1;
    }

    if (srs_rtmp_handshake(hdl) != 0) {
        //srs_human_trace("simple handshake failed.");
        logPrintWarn("PVS VSTREAM srs rtmp, simple handshake failed, rtmp:'%s'\n", url);
        Destroy(hdl);
        return -1;
    }

    //srs_human_trace("simple handshake success");
    logPrintInfo("PVS VSTREAM srs rtmp, simple handshake success, rtmp:'%s'\n", url);

    if (srs_rtmp_connect_app(hdl) != 0) {
        //srs_human_trace("connect vhost/app failed.");
        logPrintWarn("PVS VSTREAM srs rtmp, connect vhost/app failed, rtmp:'%s'\n", url);
        Destroy(hdl);
        return -1;
    }
    //srs_human_trace("connect vhost/app success");
    logPrintInfo("PVS VSTREAM srs rtmp, connect vhost/app success, rtmp:'%s'\n", url);

    if (srs_rtmp_publish_stream(hdl) != 0) {
        //srs_human_trace("publish stream failed.");
        logPrintWarn("PVS VSTREAM srs rtmp, publish stream failed, rtmp:'%s'\n", url);
        Destroy(hdl);
        return -1;
    }
    //srs_human_trace("publish stream success");
    logPrintInfo("PVS VSTREAM srs rtmp, publish stream success, rtmp:'%s'\n", url);

    *p_rtmp_hdl = hdl;
    return 0;
}

void SrsRtmp::Destroy(srs_rtmp_t hdl)
{
    if (!hdl)
        return;
    srs_rtmp_destroy(hdl);
}



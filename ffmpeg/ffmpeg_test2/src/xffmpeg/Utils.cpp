#include "Utils.h"



namespace utils
{

#if 0
void avpacket_to_string(const AVPacket* pkt, std::ostringstream& ostm)
{
    std::string tab = "     ";
    std::string crlf = "\n";
    ostm << "AVPacket:\n"
        << tab << "pts:         " << pkt->pts << crlf
        << tab << "dts:         " << pkt->dts << crlf
        << tab << "size:        " << pkt->size << crlf
        << tab << "stream_idx:  " << pkt->stream_index << crlf
        << tab << "flags:       " << pkt->flags << crlf
        << tab << "side_de:     " << pkt->side_data_elems << crlf
        << tab << "duration:    " << pkt->duration << crlf
        << tab << "pos:         " << pkt->pos << crlf
        ;
}

void avctx_to_string(const AVCodecContext* ctx, std::ostringstream& ostm)
{
    std::string tab = "     ";
    std::string crlf = "\n";
    ostm << "AVCodecContext:" << crlf
        << tab << "codec_tag:         " << ctx->codec_tag << crlf
        << tab << "bit_rate:         " << ctx->bit_rate << crlf
        << tab << "bit_rate_tolerance:        " << ctx->bit_rate_tolerance << crlf
        << tab << "global_quality:  " << ctx->global_quality << crlf
        << tab << "compression_level:       " << ctx->compression_level << crlf
        << tab << "ticks_per_frame:     " << ctx->ticks_per_frame << crlf
        << tab << "width:    " << ctx->width << crlf
        << tab << "height:         " << ctx->height << crlf
        << tab << "coded_width:    " << ctx->coded_width << crlf
        << tab << "coded_height:    " << ctx->coded_height << crlf

        << tab << "gop_size:    " << ctx->gop_size << crlf
        << tab << "pix_fmt:    " << ctx->pix_fmt << crlf
        ;

}
#else

void avpacket_to_string(const AVPacket* pkt, std::ostringstream& ostm)
{
    std::string tab = "     ";
    std::string crlf = " ";
    ostm << "AVPacket:\n"
        << tab << "pts:         " << pkt->pts << crlf
        << tab << "dts:         " << pkt->dts << crlf
        << tab << "size:        " << pkt->size << crlf
        << tab << "stream_idx:  " << pkt->stream_index << crlf
        << tab << "flags:       " << pkt->flags << crlf
        << tab << "side_de:     " << pkt->side_data_elems << crlf
        << tab << "duration:    " << pkt->duration << crlf
        << tab << "pos:         " << pkt->pos << crlf
        ;
}

void avctx_to_string(const AVCodecContext* ctx, std::ostringstream& ostm)
{
    std::string tab = "     ";
    std::string crlf = " ";
    ostm << "AVCodecContext:" << crlf
        << tab << "codec_tag:         " << ctx->codec_tag << crlf
        << tab << "bit_rate:         " << ctx->bit_rate << crlf
        << tab << "bit_rate_tolerance:        " << ctx->bit_rate_tolerance << crlf
        << tab << "global_quality:  " << ctx->global_quality << crlf
        << tab << "compression_level:       " << ctx->compression_level << crlf
        << tab << "ticks_per_frame:     " << ctx->ticks_per_frame << crlf
        << tab << "width:    " << ctx->width << crlf
        << tab << "height:         " << ctx->height << crlf
        << tab << "coded_width:    " << ctx->coded_width << crlf
        << tab << "coded_height:    " << ctx->coded_height << crlf

        << tab << "gop_size:    " << ctx->gop_size << crlf
        << tab << "pix_fmt:    " << ctx->pix_fmt << crlf
        ;

}

#endif

std::string picture_type(int v)
{
    switch (v)
    {
    case AV_PICTURE_TYPE_NONE: return "Undefined";
    case AV_PICTURE_TYPE_I: return "Intra";
    case AV_PICTURE_TYPE_P: return "Predicted";
    case AV_PICTURE_TYPE_B: return "Bi-dir predicted";
    case AV_PICTURE_TYPE_S: return "S(GMC)-VOP MPEG-4";
    case AV_PICTURE_TYPE_SI: return "Switching Intra";
    case AV_PICTURE_TYPE_SP: return "Switching Predicted";
    case AV_PICTURE_TYPE_BI: return "BI type";
    default:
        break;
    }
    return "??";
}

}





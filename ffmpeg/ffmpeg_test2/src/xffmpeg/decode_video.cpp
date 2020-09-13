/*
 * Copyright (c) 2001 Fabrice Bellard
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * @file
 * video decoding with libavcodec API example
 *
 * @example decode_video.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sstream>
#include <iostream>
#include <cassert>

#include "xffmpeg.h"

#include "Utils.h"
#include "FFmpegUtils.h"

//#define INBUF_SIZE 4096
// #define INBUF_SIZE 1000
#define INBUF_SIZE 1073741824

//#define XEXIT(x) exit(1)
#define XEXIT(x) (assert(0))

static void pgm_save(unsigned char *buf, int wrap, int xsize, int ysize,
                     char *filename)
{
    FILE *f;
    int i;

    f = fopen(filename,"w");
    fprintf(f, "P5\n%d %d\n%d\n", xsize, ysize, 255);
    for (i = 0; i < ysize; i++)
        fwrite(buf + i * wrap, 1, xsize, f);
    fclose(f);
}

static void decode(AVCodecContext *dec_ctx, AVFrame *frame, AVPacket *pkt,
                   const char *filename)
{
    char buf[1024];
    int ret;

    ret = avcodec_send_packet(dec_ctx, pkt);
    if (ret < 0) {
        fprintf(stderr, "Error sending a packet for decoding\n");
        XEXIT(0);
    }

    std::ostringstream ostm;
    utils::avctx_to_string(dec_ctx, ostm);
    //std::cout << ostm.str();

    while (ret >= 0) {
        //AV_PIX_FMT_YUV420P;
        //avcodec_decode_video2();
        // av_read_frame();
        ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
            fprintf(stderr, "Error during decoding\n");
            XEXIT(1);
        }

        //printf("saving frame %3d\n", dec_ctx->frame_number);
        fflush(stdout);

        /* the picture is allocated by the decoder. no need to
           free it */
        snprintf(buf, sizeof(buf), "%s-%d.jpeg", filename, dec_ctx->frame_number);
        //pgm_save(frame->data[0], frame->linesize[0], frame->width, frame->height, buf);
#if 1
        if (frame->key_frame == 1) {
            //pgm_save(frame->data[0], frame->linesize[0], frame->width, frame->height, buf);
            //printf("ctx_fnumber: %d %d %d    [%d %d %d]\n", dec_ctx->frame_number, frame->key_frame, frame->pts , frame->linesize[0], frame->width, frame->height );
            ffmpeg_util_save_jpeg(frame, buf);
        }
#endif
    std::cout << "pic_type: " << utils::picture_type(frame->pict_type) 
        << " key_frame: " << frame->key_frame
        << " pic_frame_num: " << frame->coded_picture_number
        << " frame_number: " << dec_ctx->frame_number
        << " pts: " << frame->pts
        << " [" << frame->width << "," << frame->height
        << "\n";
    }
}

int TestDecode()
{
    const char *filename, *outfilename;
    const AVCodec *codec;
    AVCodecParserContext *parser;
    AVCodecContext *c= NULL;
    FILE *f;
    AVFrame *frame;
    uint8_t* xinbuf = new uint8_t[INBUF_SIZE + AV_INPUT_BUFFER_PADDING_SIZE];
    uint8_t* inbuf = xinbuf;
    //uint8_t inbuf[INBUF_SIZE + AV_INPUT_BUFFER_PADDING_SIZE];
    uint8_t *data;
    size_t   data_size;
    int ret;
    AVPacket *pkt;

#if 0
    if (argc <= 2) {
        fprintf(stderr, "Usage: %s <input file> <output file>\n"
                "And check your input file is encoded by mpeg1video please.\n", argv[0]);
        XEXIT(1);
    }
#endif

    filename = "E:/resource/xiaoen.h264";
    outfilename = "b";

    pkt = av_packet_alloc();
    if (!pkt)
        XEXIT(1);

    /* set end of buffer to 0 (this ensures that no overreading happens for damaged MPEG streams) */
    memset(inbuf + INBUF_SIZE, 0, AV_INPUT_BUFFER_PADDING_SIZE);

    /* find the MPEG-1 video decoder */
    //codec = avcodec_find_decoder(AV_CODEC_ID_MPEG1VIDEO);
    codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec) {
        fprintf(stderr, "Codec not found\n");
        XEXIT(1);
    }

    parser = av_parser_init(codec->id);
    if (!parser) {
        fprintf(stderr, "parser not found\n");
        XEXIT(1);
    }

    c = avcodec_alloc_context3(codec);
    if (!c) {
        fprintf(stderr, "Could not allocate video codec context\n");
        XEXIT(1);
    }

    /* For some codecs, such as msmpeg4 and mpeg4, width and height
       MUST be initialized there because this information is not
       available in the bitstream. */

    /* open it */
    if (avcodec_open2(c, codec, NULL) < 0) {
        fprintf(stderr, "Could not open codec\n");
        XEXIT(1);
    }

    f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Could not open %s\n", filename);
        XEXIT(1);
    }

    frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Could not allocate video frame\n");
        XEXIT(1);
    }

    int cnt = 0;
    int total = 0;
    while (!feof(f)) {
        /* read raw data from the input file */
        data_size = fread(inbuf, 1, INBUF_SIZE, f);
        if (!data_size)
            break;
        total += data_size;

        /* use the parser to split the data into frames */
        data = inbuf;
        while (data_size > 0) {
            ret = av_parser_parse2(parser, c, &pkt->data, &pkt->size,
                                   data, data_size, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
            if (ret < 0) {
                fprintf(stderr, "Error while parsing\n");
                XEXIT(1);
            }
            data      += ret;
            data_size -= ret;
            if (pkt->size) {
                ++cnt;
                std::ostringstream ostm;
                utils::avpacket_to_string(pkt, ostm);
                //std::cout << ostm.str();

                //decode(c, frame, pkt, outfilename);
                //XEXIT(1);
            }
        }
    }

    /* flush the decoder */
    decode(c, frame, NULL, outfilename);

    fclose(f);

    av_parser_close(parser);
    avcodec_free_context(&c);
    av_frame_free(&frame);
    av_packet_free(&pkt);

    std::cout << "exit success\n";
    return 0;
}

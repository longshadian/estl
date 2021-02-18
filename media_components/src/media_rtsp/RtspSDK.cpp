#include <string>
#include <iostream>
#include <sstream>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <cstring>

#include "RtspSDK.h"

#include <liveMedia.hh>
#include <BasicUsageEnvironment.hh>
#include <UsageEnvironment.hh>

namespace media
{

struct RtspRawFrame
{
    RtspRawFrameInfo info_{};
    std::vector<std::uint8_t> data_{};

    void Append(const void* data, std::size_t len)
    {
        const std::uint8_t* p = static_cast<const std::uint8_t*>(data);
        data_.assign(p, p + len);
    }
};

static std::string MediaSubsession_to_string(const MediaSubsession& subs)
{
    static const char* TAB = "\t";
    static const char* CRLF = "\n";

    std::ostringstream ostm{};
    ostm 
        << "\n\n>>>>>>>>>>>>>\n"
        << TAB << "clientPortNum:      " << subs.clientPortNum() << CRLF
        << TAB << "rtpPayloadFormat:   " << subs.rtpPayloadFormat() << CRLF
        << TAB << "savedSDPLines:      " << subs.savedSDPLines() << CRLF
        << TAB << "mediumName:         " << subs.mediumName() << CRLF
        << TAB << "codecName:          " << subs.codecName() << CRLF
        << TAB << "protocolName:       " << subs.protocolName() << CRLF
        << TAB << "controlPath:        " << subs.controlPath() << CRLF
        << TAB << "isSSM:              " << static_cast<int>(subs.isSSM()) << CRLF
        << TAB << "videoWidth:         " << subs.videoWidth() << CRLF
        << TAB << "videoHeight:        " << subs.videoHeight() << CRLF
        << TAB << "videoFPS:           " << subs.videoFPS() << CRLF
        << TAB << "numChannels:        " << subs.numChannels() << CRLF
        ;
    return ostm.str();
}

static bool StringCaseEqual(const char* p1, const char* p2)
{
    auto len1 = std::strlen(p1);
    auto len2 = std::strlen(p2);
    if (len1 != len2)
        return false;
    return strncasecmp(p1, p2, len1) == 0;
}

using Live555_FrameCallback = std::function<void(MediaSubsession* s,
    void* frame, unsigned frameSize, unsigned numTruncatedBytes,
    struct timeval presentationTime, unsigned durationInMicroseconds)>;

static void continueAfterDESCRIBE(RTSPClient* rtspClient, int resultCode, char* resultString);
static void continueAfterSETUP(RTSPClient* rtspClient, int resultCode, char* resultString);
static void continueAfterPLAY(RTSPClient* rtspClient, int resultCode, char* resultString);

// called when a stream's subsession (e.g., audio or video substream) ends
static void subsessionAfterPlaying(void* clientData);

// called when a RTCP "BYE" is received for a subsession
static void subsessionByeHandler(void* clientData/*, char const* reason*/);

// called at the end of a stream's expected duration (if the stream has not already signaled its end using a RTCP "BYE")
static void streamTimerHandler(void* clientData);

// Used to iterate through each stream's 'subsessions', setting up each one:
static void setupNextSubsession(RTSPClient* rtspClient);

// Used to shut down and close a stream (including its "RTSPClient" object):
static void shutdownStream(RTSPClient* rtspClient, int exitCode = 1);

// A function that outputs a string that identifies each stream (for debugging output).  Modify this if you wish:
static
UsageEnvironment& operator<<(UsageEnvironment& env, const RTSPClient& rtspClient) {
    return env << "[URL:\"" << rtspClient.url() << "\"]: ";
}

// A function that outputs a string that identifies each subsession (for debugging output).  Modify this if you wish:
static
UsageEnvironment& operator<<(UsageEnvironment& env, const MediaSubsession& subsession) {
    return env << subsession.mediumName() << "/" << subsession.codecName();
}


class StreamClientState {
public:
    StreamClientState();
    virtual ~StreamClientState();

public:
    MediaSubsessionIterator* iter;
    MediaSession* session;
    MediaSubsession* subsession;
    TaskToken streamTimerTask;
    double duration;
};

// If you're streaming just a single stream (i.e., just from a single URL, once), then you can define and use just a single
// "StreamClientState" structure, as a global variable in your application.  However, because - in this demo application - we're
// showing how to play multiple streams, concurrently, we can't do that.  Instead, we have to have a separate "StreamClientState"
// structure for each "RTSPClient".  To do this, we subclass "RTSPClient", and add a "StreamClientState" field to the subclass:

class ourRTSPClient : public RTSPClient {
public:
    static ourRTSPClient* createNew(UsageEnvironment& env, char const* rtspURL,
        int verbosityLevel = 0,
        char const* applicationName = NULL,
        portNumBits tunnelOverHTTPPortNum = 0);

protected:
    ourRTSPClient(UsageEnvironment& env, char const* rtspURL,
        int verbosityLevel, char const* applicationName, portNumBits tunnelOverHTTPPortNum);
    // called only by createNew();
    virtual ~ourRTSPClient();

public:
    StreamClientState scs;
    Live555_FrameCallback proc_;
    bool shutdown_;
    int over_tcp_;
};

// Define a data sink (a subclass of "MediaSink") to receive the data for each subsession (i.e., each audio or video 'substream').
// In practice, this might be a class (or a chain of classes) that decodes and then renders the incoming audio or video.
// Or it might be a "FileSink", for outputting the received data into a file (as is done by the "openRTSP" application).
// In this example code, however, we define a simple 'dummy' sink that receives incoming data, but does nothing with it.

class DummySink : public MediaSink {
public:
    static DummySink* createNew(UsageEnvironment& env,
        MediaSubsession& subsession, // identifies the kind of data that's being received
        char const* streamId = NULL); // identifies the stream itself (optional)

private:
    DummySink(UsageEnvironment& env, MediaSubsession& subsession, char const* streamId);
    // called only by "createNew()"
    virtual ~DummySink();

    static void afterGettingFrame(void* clientData, unsigned frameSize,
        unsigned numTruncatedBytes,
        struct timeval presentationTime,
        unsigned durationInMicroseconds);
    void afterGettingFrame(unsigned frameSize, unsigned numTruncatedBytes,
        struct timeval presentationTime, unsigned durationInMicroseconds);

private:
    // redefined virtual functions:
    virtual Boolean continuePlaying();

private:
    u_int8_t* fReceiveBuffer;
    MediaSubsession& fSubsession;
    char* fStreamId;

public:
    Live555_FrameCallback proc_;
};


void continueAfterDESCRIBE(RTSPClient* rtspClient, int resultCode, char* resultString) {
    do {
        UsageEnvironment& env = rtspClient->envir(); // alias
        StreamClientState& scs = ((ourRTSPClient*)rtspClient)->scs; // alias

        if (resultCode != 0) {
            env << *rtspClient << "Failed to get a SDP description: " << resultString << "\n";
            delete[] resultString;
            break;
        }

        char* const sdpDescription = resultString;
        env << *rtspClient << "Got a SDP description:\n" << sdpDescription << "\n";

        // Create a media session object from this SDP description:
        scs.session = MediaSession::createNew(env, sdpDescription);
        delete[] sdpDescription; // because we don't need it anymore
        if (scs.session == NULL) {
            env << *rtspClient << "Failed to create a MediaSession object from the SDP description: " << env.getResultMsg() << "\n";
            break;
        }
        else if (!scs.session->hasSubsessions()) {
            env << *rtspClient << "This session has no media subsessions (i.e., no \"m=\" lines)\n";
            break;
        }

        // Then, create and set up our data source objects for the session.  We do this by iterating over the session's 'subsessions',
        // calling "MediaSubsession::initiate()", and then sending a RTSP "SETUP" command, on each one.
        // (Each 'subsession' will have its own data source.)
        scs.iter = new MediaSubsessionIterator(*scs.session);
        setupNextSubsession(rtspClient);
        return;
    } while (0);

    // An unrecoverable error occurred with this stream.
    shutdownStream(rtspClient);
}

// By default, we request that the server stream its data using RTP/UDP.
// If, instead, you want to request that the server stream via RTP-over-TCP, change the following to True:
//#define REQUEST_STREAMING_OVER_TCP False
#define REQUEST_STREAMING_OVER_TCP True

void setupNextSubsession(RTSPClient* rtspClient) {
    UsageEnvironment& env = rtspClient->envir(); // alias
    ourRTSPClient* our_rtsp_client = (ourRTSPClient*)rtspClient;
    StreamClientState& scs = our_rtsp_client->scs; // alias

    scs.subsession = scs.iter->next();
    if (scs.subsession != NULL) {
        if (!scs.subsession->initiate()) {
            env << *rtspClient << "Failed to initiate the \"" << *scs.subsession << "\" subsession: " << env.getResultMsg() << "\n";
            setupNextSubsession(rtspClient); // give up on this subsession; go to the next one
        }
        else {
            env << *rtspClient << "Initiated the \"" << *scs.subsession << "\" subsession (";
            if (scs.subsession->rtcpIsMuxed()) {
                env << "client port " << scs.subsession->clientPortNum();
            }
            else {
                env << "client ports " << scs.subsession->clientPortNum() << "-" << scs.subsession->clientPortNum() + 1;
            }
            env << ")\n";

            // Continue setting up this subsession, by sending a RTSP "SETUP" command:
            //rtspClient->sendSetupCommand(*scs.subsession, continueAfterSETUP, False, REQUEST_STREAMING_OVER_TCP);
            rtspClient->sendSetupCommand(*scs.subsession, continueAfterSETUP, False, our_rtsp_client->over_tcp_);
        }
        return;
    }

    // We've finished setting up all of the subsessions.  Now, send a RTSP "PLAY" command to start the streaming:
    if (scs.session->absStartTime() != NULL) {
        // Special case: The stream is indexed by 'absolute' time, so send an appropriate "PLAY" command:
        rtspClient->sendPlayCommand(*scs.session, continueAfterPLAY, scs.session->absStartTime(), scs.session->absEndTime());
    }
    else {
        scs.duration = scs.session->playEndTime() - scs.session->playStartTime();
        rtspClient->sendPlayCommand(*scs.session, continueAfterPLAY);
    }
}

void continueAfterSETUP(RTSPClient* rtspClient, int resultCode, char* resultString) {
    do {
        UsageEnvironment& env = rtspClient->envir(); // alias
        StreamClientState& scs = ((ourRTSPClient*)rtspClient)->scs; // alias

        if (resultCode != 0) {
            env << *rtspClient << "Failed to set up the \"" << *scs.subsession << "\" subsession: " << resultString << "\n";
            break;
        }

        env << *rtspClient << "Set up the \"" << *scs.subsession << "\" subsession (";
        if (scs.subsession->rtcpIsMuxed()) {
            env << "client port " << scs.subsession->clientPortNum();
        }
        else {
            env << "client ports " << scs.subsession->clientPortNum() << "-" << scs.subsession->clientPortNum() + 1;
        }
        env << ")\n";

        // Having successfully setup the subsession, create a data sink for it, and call "startPlaying()" on it.
        // (This will prepare the data sink to receive data; the actual flow of data from the client won't start happening until later,
        // after we've sent a RTSP "PLAY" command.)

        scs.subsession->sink = DummySink::createNew(env, *scs.subsession, rtspClient->url());
        // perhaps use your own custom "MediaSink" subclass instead
        if (scs.subsession->sink == NULL) {
            env << *rtspClient << "Failed to create a data sink for the \"" << *scs.subsession
                << "\" subsession: " << env.getResultMsg() << "\n";
            break;
        }

        {
            ourRTSPClient* pc = (ourRTSPClient*)rtspClient;
            DummySink* psink = (DummySink*)(scs.subsession->sink);
            psink->proc_ = pc->proc_;
        }

        env << *rtspClient << "Created a data sink for the \"" << *scs.subsession << "\" subsession\n";
        scs.subsession->miscPtr = rtspClient; // a hack to let subsession handler functions get the "RTSPClient" from the subsession 
        scs.subsession->sink->startPlaying(*(scs.subsession->readSource()),
            subsessionAfterPlaying, scs.subsession);
        // Also set a handler to be called if a RTCP "BYE" arrives for this subsession:
        if (scs.subsession->rtcpInstance() != NULL) {
            //scs.subsession->rtcpInstance()->setByeWithReasonHandler(subsessionByeHandler, scs.subsession);
            scs.subsession->rtcpInstance()->setByeHandler(subsessionByeHandler, scs.subsession);
        }
    } while (0);
    delete[] resultString;

    // Set up the next subsession, if any:
    setupNextSubsession(rtspClient);
}

void continueAfterPLAY(RTSPClient* rtspClient, int resultCode, char* resultString) {
    Boolean success = False;

    do {
        UsageEnvironment& env = rtspClient->envir(); // alias
        StreamClientState& scs = ((ourRTSPClient*)rtspClient)->scs; // alias

        if (resultCode != 0) {
            env << *rtspClient << "Failed to start playing session: " << resultString << "\n";
            break;
        }

        // Set a timer to be handled at the end of the stream's expected duration (if the stream does not already signal its end
        // using a RTCP "BYE").  This is optional.  If, instead, you want to keep the stream active - e.g., so you can later
        // 'seek' back within it and do another RTSP "PLAY" - then you can omit this code.
        // (Alternatively, if you don't want to receive the entire stream, you could set this timer for some shorter value.)
        if (scs.duration > 0) {
            unsigned const delaySlop = 2; // number of seconds extra to delay, after the stream's expected duration.  (This is optional.)
            scs.duration += delaySlop;
            unsigned uSecsToDelay = (unsigned)(scs.duration * 1000000);
            scs.streamTimerTask = env.taskScheduler().scheduleDelayedTask(uSecsToDelay, (TaskFunc*)streamTimerHandler, rtspClient);
        }

        env << *rtspClient << "Started playing session";
        if (scs.duration > 0) {
            env << " (for up to " << scs.duration << " seconds)";
        }
        env << "...\n";

        success = True;
    } while (0);
    delete[] resultString;

    if (!success) {
        // An unrecoverable error occurred with this stream.
        shutdownStream(rtspClient);
    }
}

void subsessionAfterPlaying(void* clientData) {
  MediaSubsession* subsession = (MediaSubsession*)clientData;
  RTSPClient* rtspClient = (RTSPClient*)(subsession->miscPtr);

  // Begin by closing this subsession's stream:
  Medium::close(subsession->sink);
  subsession->sink = NULL;

  // Next, check whether *all* subsessions' streams have now been closed:
  MediaSession& session = subsession->parentSession();
  MediaSubsessionIterator iter(session);
  while ((subsession = iter.next()) != NULL) {
    if (subsession->sink != NULL) return; // this subsession is still active
  }

  // All subsessions' streams have now been closed, so shutdown the client:
  shutdownStream(rtspClient);
}

void subsessionByeHandler(void* clientData/*, char const* reason*/) {
  MediaSubsession* subsession = (MediaSubsession*)clientData;
  RTSPClient* rtspClient = (RTSPClient*)subsession->miscPtr;
  UsageEnvironment& env = rtspClient->envir(); // alias

  env << *rtspClient << "Received RTCP \"BYE\"";
#if 0
  if (reason != NULL) {
    env << " (reason:\"" << reason << "\")";
    delete[] (char*)reason;
  }
#endif
  env << " on \"" << *subsession << "\" subsession\n";

  // Now act as if the subsession had closed:
  subsessionAfterPlaying(subsession);
}

void streamTimerHandler(void* clientData) {
  ourRTSPClient* rtspClient = (ourRTSPClient*)clientData;
  StreamClientState& scs = rtspClient->scs; // alias

  scs.streamTimerTask = NULL;

  // Shut down the stream:
  shutdownStream(rtspClient);
}

void shutdownStream(RTSPClient* rtspClient, int exitCode) {
    (void)exitCode;
  UsageEnvironment& env = rtspClient->envir(); // alias
  ourRTSPClient* ourClient = (ourRTSPClient*)rtspClient;
  if (ourClient->shutdown_)
      return;
  ourClient->shutdown_ = true;
  StreamClientState& scs = ourClient->scs; // alias

  // First, check whether any subsessions have still to be closed:
  if (scs.session != NULL) { 
    Boolean someSubsessionsWereActive = False;
    MediaSubsessionIterator iter(*scs.session);
    MediaSubsession* subsession;

    while ((subsession = iter.next()) != NULL) {
      if (subsession->sink != NULL) {
	Medium::close(subsession->sink);
	subsession->sink = NULL;

	if (subsession->rtcpInstance() != NULL) {
	  subsession->rtcpInstance()->setByeHandler(NULL, NULL); // in case the server sends a RTCP "BYE" while handling "TEARDOWN"
	}

	someSubsessionsWereActive = True;
      }
    }

    if (someSubsessionsWereActive) {
      // Send a RTSP "TEARDOWN" command, to tell the server to shutdown the stream.
      // Don't bother handling the response to the "TEARDOWN".
      rtspClient->sendTeardownCommand(*scs.session, NULL);
    }
  }

  env << *rtspClient << "Closing the stream.\n";
  Medium::close(rtspClient);
    // Note that this will also cause this stream's "StreamClientState" structure to get reclaimed.

#if 0
  if (--rtspClientCount == 0) {
    // The final stream has ended, so exit the application now.
    // (Of course, if you're embedding this code into your own application, you might want to comment this out,
    // and replace it with "eventLoopWatchVariable = 1;", so that we leave the LIVE555 event loop, and continue running "main()".)
    exit(exitCode);
  }
#endif

}



// Implementation of "ourRTSPClient":

ourRTSPClient* ourRTSPClient::createNew(UsageEnvironment& env, char const* rtspURL,
    int verbosityLevel, char const* applicationName, portNumBits tunnelOverHTTPPortNum) {
    return new ourRTSPClient(env, rtspURL, verbosityLevel, applicationName, tunnelOverHTTPPortNum);
}

ourRTSPClient::ourRTSPClient(UsageEnvironment& env, char const* rtspURL,
    int verbosityLevel, char const* applicationName, portNumBits tunnelOverHTTPPortNum)
    : RTSPClient(env, rtspURL, verbosityLevel, applicationName, tunnelOverHTTPPortNum, -1) {
    over_tcp_ = 0;
}

ourRTSPClient::~ourRTSPClient() {
}


// Implementation of "StreamClientState":

StreamClientState::StreamClientState()
    : iter(NULL), session(NULL), subsession(NULL), streamTimerTask(NULL), duration(0.0) {
}

StreamClientState::~StreamClientState() {
    delete iter;
    if (session != NULL) {
        // We also need to delete "session", and unschedule "streamTimerTask" (if set)
        UsageEnvironment& env = session->envir(); // alias

        env.taskScheduler().unscheduleDelayedTask(streamTimerTask);
        Medium::close(session);
    }
}

// Implementation of "DummySink":

// Even though we're not going to be doing anything with the incoming data, we still need to receive it.
// Define the size of the buffer that we'll use:
#define DUMMY_SINK_RECEIVE_BUFFER_SIZE 10000000

DummySink* DummySink::createNew(UsageEnvironment& env, MediaSubsession& subsession, char const* streamId) {
    return new DummySink(env, subsession, streamId);
}

DummySink::DummySink(UsageEnvironment& env, MediaSubsession& subsession, char const* streamId)
    : MediaSink(env),
    fSubsession(subsession) {
    fStreamId = strDup(streamId);
    fReceiveBuffer = new u_int8_t[DUMMY_SINK_RECEIVE_BUFFER_SIZE];
    fReceiveBuffer[0] = 0;
    fReceiveBuffer[1] = 0;
    fReceiveBuffer[2] = 0;
    fReceiveBuffer[3] = 1;
    proc_ = {};
}

DummySink::~DummySink() {
    delete[] fReceiveBuffer;
    delete[] fStreamId;
}

void DummySink::afterGettingFrame(void* clientData, unsigned frameSize, unsigned numTruncatedBytes,
    struct timeval presentationTime, unsigned durationInMicroseconds) {
    DummySink* sink = (DummySink*)clientData;
    sink->afterGettingFrame(frameSize, numTruncatedBytes, presentationTime, durationInMicroseconds);
}

// If you don't want to see debugging output for each received frame, then comment out the following line:
// #define DEBUG_PRINT_EACH_RECEIVED_FRAME 1

void DummySink::afterGettingFrame(unsigned frameSize, unsigned numTruncatedBytes,
    struct timeval presentationTime, unsigned durationInMicroseconds) {
    // We've just received a frame of data.  (Optionally) print out information about it:
#ifdef DEBUG_PRINT_EACH_RECEIVED_FRAME
    if (fStreamId != NULL) envir() << "Stream \"" << fStreamId << "\"; ";
    envir() << fSubsession.mediumName() << "/" << fSubsession.codecName() << ":\tReceived " << frameSize << " bytes";
    if (numTruncatedBytes > 0) envir() << " (with " << numTruncatedBytes << " bytes truncated)";
    char uSecsStr[6 + 1]; // used to output the 'microseconds' part of the presentation time
    sprintf(uSecsStr, "%06u", (unsigned)presentationTime.tv_usec);
    envir() << ".\tPresentation time: " << (int)presentationTime.tv_sec << "." << uSecsStr;
    if (fSubsession.rtpSource() != NULL && !fSubsession.rtpSource()->hasBeenSynchronizedUsingRTCP()) {
        envir() << "!"; // mark the debugging output to indicate that this presentation time is not RTCP-synchronized
    }
#ifdef DEBUG_PRINT_NPT
    envir() << "\tNPT: " << fSubsession.getNormalPlayTime(presentationTime);
#endif
    envir() << "\n";
#endif

    if (0) {
        //std::cout << "code name: " << fSubsession.codecName() << "\n";
        //std::cout << MediaSubsession_to_string(fSubsession);
        std::ostringstream ostm{};
        uint16_t v{};
        memcpy(&v, fReceiveBuffer + 4, sizeof(v));
        ostm << int(v >> 15) << " ";
        ostm << int((v >> 9)& 0x3F) << " ";
        ostm << int((v >> 3)& 0x3F) << " ";
        ostm << int(v & 0x7) << " ";
        std::cout << "NAL: " << ostm.str() << "\n";
    }

    if (proc_) {
        //std::cout << MediaSubsession_to_string(fSubsession);
        proc_(&fSubsession, fReceiveBuffer, frameSize + 4, numTruncatedBytes, presentationTime, durationInMicroseconds);
    }

/*
    if (proc_ && strcmp(fSubsession.codecName(), "H265") == 0)
        proc_(fReceiveBuffer, frameSize + 4, numTruncatedBytes, presentationTime, durationInMicroseconds);
        //proc_(fReceiveBuffer + 4, frameSize, numTruncatedBytes, presentationTime, durationInMicroseconds);
*/

    // Then continue, to request the next frame of data:
    continuePlaying();
}

Boolean DummySink::continuePlaying() {
    if (fSource == NULL) return False; // sanity check (should not happen)

    // Request the next frame of data from our input source.  "afterGettingFrame()" will get called later, when it arrives:
    fSource->getNextFrame(fReceiveBuffer+4, DUMMY_SINK_RECEIVE_BUFFER_SIZE,
        afterGettingFrame, this,
        onSourceClosure, this);
    return True;
}


class RtspClientImpl;
class RtspSDK::RtspSDKImpl
{
public:
    using RtspTask = std::function<void()>;

    RtspSDKImpl();
    ~RtspSDKImpl();

    int Init();
    void Cleanup();
    int StartPullRtsp(const RtspParam* param, std::string url, FrameCallback user_cb, RtspHandle* hdl);
    int StopPullRtsp(RtspHandle hdl);
    void PostTask(RtspTask t);
    void Async_ClientReceivedFrame(RtspHandle hdl, std::shared_ptr<RtspRawFrame> frame);
    std::unique_ptr<RtspRawFrame> CreateRtspRawFrame();
private:
    void ThreadRun();
    void Async_CreateNewClient(RtspHandle hdl, std::shared_ptr<RtspParam> param, std::string url, FrameCallback cb);
    void ClientStopPull(RtspHandle hdl);
    void OnCreateNewClient(RtspHandle new_hdl, std::shared_ptr<RtspParam> param, std::string url, FrameCallback user_cb);
    void OnRawFrameReceived(RtspHandle hdl, std::shared_ptr<RtspRawFrame> frame);
    RtspClientImpl* FindClient(RtspHandle hdl);

private:
    std::atomic<bool>                   running_;
    std::atomic<RtspHandle>             next_hdl_;
    std::mutex                          mtx_;
    std::condition_variable             cond_;
    std::queue<RtspTask>                queue_;
    std::unordered_map<RtspHandle, std::unique_ptr<RtspClientImpl>> clients_;
    std::thread                         thd_;
};

//=======================================================================
// class RtspClientImpl
//=======================================================================
class RtspClientImpl
{
public:
    RtspClientImpl(RtspSDK::RtspSDKImpl* sdk, RtspHandle hdl)
        : sdk_impl_{sdk}
        , hdl_{hdl}
        , param_{}
        , rtsp_url_{}
        , user_cb_{}
    {
    }

    virtual ~RtspClientImpl() = default;

    virtual int Init(std::shared_ptr<RtspParam> param, std::string url, FrameCallback cb) 
    {
        param_ = std::move(param);
        rtsp_url_ = std::move(url);
        user_cb_ = std::move(cb);
        return 0;
    }

    virtual void Loop() = 0;
    virtual void Stop() = 0;

    RtspSDK::RtspSDKImpl*       sdk_impl_{};
    RtspHandle                  hdl_{};
    std::shared_ptr<RtspParam>  param_{};
    std::string                 rtsp_url_{};
    FrameCallback               user_cb_{};
};

//=======================================================================
// class RtspClient_Live555
//=======================================================================

class RtspClient_Live555 : public RtspClientImpl
{
public:
    using Super = RtspClientImpl;

    RtspClient_Live555(RtspSDK::RtspSDKImpl* sdk, RtspHandle hdl);
    virtual ~RtspClient_Live555();

    virtual int Init(std::shared_ptr<RtspParam> param, std::string url, FrameCallback cb);
    virtual void Loop();
    virtual void Stop();

private:
    void StoreErrorInfo(int ecode, const char* emsg)
    {
        error_code_ = ecode;
        error_msg_.clear();
        if (emsg)
            error_msg_ = emsg;
    }

    void StoreErrorInfo(int ecode, std::string p)
    {
        error_code_ = ecode;
        error_msg_ = std::move(p);
    }

    static void TimeoutProc0(void* data);
    void TimeoutProc();

    void OnFrameReceived(MediaSubsession* s,
        void* frame, unsigned frameSize, unsigned numTruncatedBytes,
        struct timeval presentationTime, unsigned durationInMicroseconds);

private:
    TaskScheduler* m_scheduler_;
    UsageEnvironment* m_env_;
    ourRTSPClient* m_our_client_;
    TaskToken m_token_;

    int error_code_;
    std::string error_msg_;

    char exit_looping_;

    std::thread thd_;
};

#if 0
//=======================================================================
// class RtspClient
//=======================================================================
RtspClient::RtspClient()
    : impl_{std::make_unique<RtspClient_Live555>()}
{
}

RtspClient::~RtspClient()
{
}

int RtspClient::Init(const RtspParam* param, std::string url, FrameCallback cb)
{
    return impl_->Init(param, std::move(url), std::move(cb));
}

void RtspClient::Loop()
{
    return impl_->Loop();
}

void RtspClient::Stop()
{
    return impl_->Stop();
}
#endif

//=======================================================================
// class RtspClient_Live555
//=======================================================================

RtspClient_Live555::RtspClient_Live555(RtspSDK::RtspSDKImpl* sdk, RtspHandle hdl)
    : Super{sdk ,hdl}
    , m_scheduler_{}
    , m_env_{}
    , m_our_client_{}
    , m_token_{}
    , error_code_{}
    , error_msg_{}
    , exit_looping_{}
    , thd_{}
{
}

RtspClient_Live555::~RtspClient_Live555()
{
    Stop();
    if (thd_.joinable())
        thd_.join();
    if  (m_scheduler_)
        delete m_scheduler_;
}

int RtspClient_Live555::Init(std::shared_ptr<RtspParam> param, std::string url, FrameCallback cb)
{
    Super::Init(std::move(param), std::move(url), std::move(cb));

    m_scheduler_ = BasicTaskScheduler::createNew();
    m_env_ = BasicUsageEnvironment::createNew(*m_scheduler_);
    m_our_client_ = ourRTSPClient::createNew(*m_env_, rtsp_url_.c_str(), 0, "RtspClient");
    if (!m_our_client_) {
        StoreErrorInfo(m_env_->getErrno(), m_env_->getResultMsg());
        return -1;
    }
    using namespace std::placeholders;
    m_our_client_->proc_ = std::bind(&RtspClient_Live555::OnFrameReceived, this, _1, _2, _3, _4, _5, _6);
    m_our_client_->over_tcp_ = (param_->protocol_type == EProtocolType::TCP) ? 1 : 0;

#if 1
    // Next, send a RTSP "DESCRIBE" command, to get a SDP description for the stream.
    // Note that this command - like all RTSP commands - is sent asynchronously; we do not block, waiting for a response.
    // Instead, the following function call returns immediately, and we handle the RTSP response later, from within the event loop:
    m_our_client_->sendDescribeCommand(continueAfterDESCRIBE);
    m_token_ = m_env_->taskScheduler().scheduleDelayedTask(1000000 * 1, &RtspClient_Live555::TimeoutProc0, this);
#endif

    std::thread temp_thd{std::bind(&RtspClient_Live555::Loop, this)};
    std::swap(thd_, temp_thd);
    return 0;
}

void RtspClient_Live555::Loop()
{
    m_env_->taskScheduler().doEventLoop(&exit_looping_);
    shutdownStream(m_our_client_, 1);
}

void RtspClient_Live555::Stop()
{
    exit_looping_ = 1;
}

void RtspClient_Live555::TimeoutProc0(void* data)
{
    RtspClient_Live555* pthis = static_cast<RtspClient_Live555*>(data);
    pthis->TimeoutProc();
}

void RtspClient_Live555::TimeoutProc()
{
}

void RtspClient_Live555::OnFrameReceived(MediaSubsession* ss,
    void* frame, unsigned frameSize, unsigned numTruncatedBytes,
    struct timeval presentationTime, unsigned durationInMicroseconds)
{
    (void)numTruncatedBytes;
    (void)presentationTime;
    (void)durationInMicroseconds;

    auto pframe = sdk_impl_->CreateRtspRawFrame();
    if (StringCaseEqual(ss->codecName(), "H264")) {
        pframe->info_.codec_type = ECodecType::H264;
    } else if (StringCaseEqual(ss->codecName(), "H265")) {
        pframe->info_.codec_type = ECodecType::HEVC;
    } else {
        pframe->info_.codec_type = ECodecType::UnknownType;
    }
    pframe->info_.width = ss->videoWidth();
    pframe->info_.height = ss->videoHeight();
    pframe->Append(frame, static_cast<std::size_t>(frameSize));
    sdk_impl_->Async_ClientReceivedFrame(hdl_, std::move(pframe));
}

//=======================================================================
// class RtspSDK
//=======================================================================
RtspSDK::RtspSDK()
    : impl_{std::make_unique<RtspSDKImpl>()}
{
}

RtspSDK::~RtspSDK()
{
    impl_->Cleanup();
}

int RtspSDK::Init()
{
    return impl_->Init();
}

void RtspSDK::Cleanup()
{
    impl_->Cleanup();
}

int RtspSDK::StartPullRtsp(const RtspParam* param, std::string url, FrameCallback cb, RtspHandle* hdl)
{
    return impl_->StartPullRtsp(param, std::move(url), std::move(cb), hdl);
}

int RtspSDK::StopPullRtsp(RtspHandle hdl)
{
    return impl_->StopPullRtsp(hdl);
}

//=======================================================================
// class RtspSDKImpl
//=======================================================================
RtspSDK::RtspSDKImpl::RtspSDKImpl()
    : running_{}
    , next_hdl_{}
    , mtx_{}
    , cond_{}
    , queue_{}
    , clients_{}
    , thd_{}
{
}

RtspSDK::RtspSDKImpl::~RtspSDKImpl()
{
    Cleanup();
    if (thd_.joinable())
        thd_.join();
}

int RtspSDK::RtspSDKImpl::Init()
{
    running_ = true;
    std::thread temp_thd{
        [this] { this->ThreadRun(); }
    };
    std::swap(thd_, temp_thd);
    return 0;
}

void RtspSDK::RtspSDKImpl::Cleanup()
{
    running_ = false;
    PostTask([]{});
}

int RtspSDK::RtspSDKImpl::StartPullRtsp(const RtspParam* param, std::string url, FrameCallback user_cb, RtspHandle* hdl)
{
    auto new_hdl = ++next_hdl_;
    auto p = std::make_shared<RtspParam>(*param);
    Async_CreateNewClient(new_hdl, std::move(p), std::move(url), std::move(user_cb));
    *hdl = new_hdl;
    return 0;
}

int RtspSDK::RtspSDKImpl::StopPullRtsp(RtspHandle hdl)
{
    PostTask([this, hdl] () mutable
    {
        this->ClientStopPull(hdl);
    });
    return 0;
}

void RtspSDK::RtspSDKImpl::Async_ClientReceivedFrame(RtspHandle hdl, std::shared_ptr<RtspRawFrame> frame)
{
    PostTask([this, hdl, frame_ex = std::move(frame)]() mutable 
    {
        this->OnRawFrameReceived(hdl, std::move(frame_ex));
    });
}

std::unique_ptr<RtspRawFrame> RtspSDK::RtspSDKImpl::CreateRtspRawFrame()
{
    return std::make_unique<RtspRawFrame>();
}

void RtspSDK::RtspSDKImpl::PostTask(RtspTask t)
{
    std::lock_guard<std::mutex> lk{mtx_};
    queue_.emplace(std::move(t));
    cond_.notify_one();
}

void RtspSDK::RtspSDKImpl::ThreadRun()
{
    RtspTask task{};
    while (running_) {
        {
            std::unique_lock<std::mutex> lk{mtx_};
            cond_.wait(lk, [this]() { return !queue_.empty(); });
            task = std::move(queue_.front());
            queue_.pop();
        }
        try {
            task();
        } catch (...) {
        }
    }
}

void RtspSDK::RtspSDKImpl::Async_CreateNewClient(RtspHandle hdl, std::shared_ptr<RtspParam> param, std::string url, FrameCallback cb)
{
    PostTask([this, hdl, param_ex = std::move(param), url_ex = std::move(url), cb_ex = std::move(cb)]() mutable
    {
        this->OnCreateNewClient(hdl, std::move(param_ex), std::move(url_ex), std::move(cb_ex));
    });
}

void RtspSDK::RtspSDKImpl::ClientStopPull(RtspHandle hdl)
{
    auto* client = FindClient(hdl);
    if (!client)
        return;
    client->Stop();
    clients_.erase(hdl);
}

void RtspSDK::RtspSDKImpl::OnCreateNewClient(RtspHandle new_hdl, std::shared_ptr<RtspParam> param, std::string url, FrameCallback user_cb)
{
    auto new_client = std::make_unique<RtspClient_Live555>(this, new_hdl);
    new_client->Init(std::move(param), std::move(url), std::move(user_cb));
    clients_.emplace(new_hdl, std::move(new_client));
}

void RtspSDK::RtspSDKImpl::OnRawFrameReceived(RtspHandle hdl, std::shared_ptr<RtspRawFrame> frame)
{
    // 1. client不存在，移除此client
    // 2. client存在，执行user cb
    auto* client = FindClient(hdl); 
    if (!client) {
        client->Stop();
        clients_.erase(hdl);
        client = nullptr;
        return;
    }
    client->user_cb_(client->hdl_, &frame->info_, frame->data_.data(), static_cast<std::int32_t>(frame->data_.size()));
}

RtspClientImpl* RtspSDK::RtspSDKImpl::FindClient(RtspHandle hdl)
{
    auto it = clients_.find(hdl);
    if (it == clients_.end())
        return nullptr;
    return it->second.get();
}


} // namespace media


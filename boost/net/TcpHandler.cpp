#include "TcpHandler.h"

#include <chrono>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "TcpServer.h"
#include "TcpEvent.h"

TcpHandler::TcpHandler(boost::asio::ip::tcp::socket socket, TcpServer& server, int64_t conn_id
    , std::function<void(int64_t)> cb)
    : m_tcp_server(server)
    , m_socket(std::move(socket))
    , m_is_closed(false)
    , m_conn_id(conn_id)
    , m_read_fix_buffer()
    , m_read_total_buffer()
    , m_write_buffer()
    , m_server_cb(std::move(cb))
{
}

TcpHandler::~TcpHandler()
{
}

void TcpHandler::Init()
{
    DoRead();
}

void TcpHandler::Send(std::shared_ptr<ByteBuffer> buffer)
{
    GetIOContext().post([this, self = shared_from_this(), buffer]()
        {
            bool wait_write = !m_write_buffer.empty();
            m_write_buffer.push_back(std::move(buffer));
            if (!wait_write) {
                DoWrite();
            }
        });
}

void TcpHandler::DoWrite()
{
    boost::asio::async_write(m_socket
        , boost::asio::buffer(m_write_buffer.front()->GetReaderPtr(), m_write_buffer.front()->ReadableSize())
        , std::bind(&TcpHandler::WriteCallback, shared_from_this(), std::placeholders::_1, std::placeholders::_2));
}

void TcpHandler::WriteCallback(boost::system::error_code ec, std::size_t length)
{
    (void)length;
    if (ec) {
        DoClosed(CLOSED_TYPE::NORMAL);
        return;
    }
    m_write_buffer.pop_front();
    if (!m_write_buffer.empty()) {
        DoWrite();
    }
}

void TcpHandler::DoRead()
{
    std::shared_ptr<boost::asio::deadline_timer> timer{};
    if (GetTimeoutTime() > 0)
        timer = SetTimeoutTimer(GetTimeoutTime());

    m_socket.async_read_some(boost::asio::buffer(m_read_fix_buffer)
        , [this, self = shared_from_this(), timer](boost::system::error_code ec, size_t length)
        {
            TimeoutCancel(timer);
            if (ec) {
                DoClosed();
                return;
            }

            m_read_total_buffer.Append(m_read_fix_buffer.data(), length);
            if (Decode()) {
                DoRead();
            } else {
                DoClosed();
            }
        });
}

boost::asio::ip::tcp::socket& TcpHandler::GetSocket()
{
    return m_socket;
}

void TcpHandler::DoClosed(CLOSED_TYPE type)
{
    if (m_is_closed.exchange(true))
        return;
    if (type == CLOSED_TYPE::NORMAL) {
        m_tcp_server.GetEvent().OnClosed(GetHdl());
    } else if (type == CLOSED_TYPE::TIMEOUT) {
        m_tcp_server.GetEvent().OnTimeout(GetHdl());
    }
    boost::system::error_code ec;
    m_socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
    m_socket.close(ec);
    m_server_cb(GetConnID());
}

void TcpHandler::Shutdown()
{
    if (m_is_closed)
        return;
    GetIOContext().post([self = shared_from_this()]() 
        {
            self->DoClosed(CLOSED_TYPE::ACTIVITY);
        });
}

TcpHandler::DeadlineTimerPtr TcpHandler::SetTimeoutTimer(size_t seconds)
{
    auto timer = std::make_shared<boost::asio::deadline_timer>(GetIOContext());
    timer->expires_from_now(boost::posix_time::seconds(seconds));
    timer->async_wait([self = shared_from_this()](const boost::system::error_code& ec) {
        if (!ec) {
            self->DoClosed(CLOSED_TYPE::TIMEOUT);
        }
    });
    return timer;
}

void TcpHandler::TimeoutCancel(DeadlineTimerPtr timer)
{
    if (timer) {
        boost::system::error_code ec;
        timer->cancel(ec);
    }
}

boost::asio::io_service& TcpHandler::GetIOContext()
{
    return m_socket.get_io_service();
}

TcpHdl TcpHandler::GetHdl()
{
    return TcpHdl{shared_from_this()};
}

int64_t TcpHandler::GetConnID() const
{
    return m_conn_id;
}

size_t TcpHandler::GetTimeoutTime() const
{
    return m_tcp_server.GetOption().m_timeout_seconds;
}

bool TcpHandler::Decode()
{
    m_tcp_server.GetEvent().OnDecode(GetHdl(), m_read_total_buffer);
    m_read_total_buffer.MemoryMove();
    return true;
}

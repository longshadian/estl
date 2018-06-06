#include "TcpServer.h"

#include "TcpHandler.h"
#include "Message.h"

TcpServer::TcpServer(uint16_t port, std::unique_ptr<TcpServerEvent> evt, Option option)
    : m_is_running()
    , m_thread()
    , m_io_context()
    , m_work()
    , m_acceptor(m_io_context, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port))
    , m_socket(m_io_context)
    , m_handlers()
    , m_option(option)
    , m_event(std::move(evt))
    , m_handler_index()
{
}

TcpServer::~TcpServer()
{
    Stop();
    if (m_thread.joinable())
        m_thread.join();
}

void TcpServer::Start()
{
    if (m_is_running.exchange(true))
        return;
    m_work = std::make_unique<boost::asio::io_service::work>(m_io_context);
    DoAccept();
    std::thread temp{std::bind(&TcpServer::Run, this)};
    m_thread = std::move(temp);
}

void TcpServer::Stop()
{
    if (!m_is_running.exchange(false)) {
        return;
    }
    StopAccept();
    StopAllHandler();
    m_io_context.stop();
}

void TcpServer::WaitThreadExit()
{
    Stop();
    if (m_thread.joinable())
        m_thread.join();
}

void TcpServer::StopAccept()
{
    boost::system::error_code ec;
    m_acceptor.cancel(ec);
    m_acceptor.close(ec);
}

void TcpServer::StopAllHandler()
{
    // TODO
}

void TcpServer::DoAccept()
{
    m_acceptor.async_accept(m_socket
        , [this](const boost::system::error_code& ec)
        {
            if (!ec) {
                auto handler = CreateHandler();
                if (handler) {
                    m_event->OnAccept(handler->GetHdl());
                } else {
                    // refulse accept
                    auto this_socket = std::move(m_socket);
                    boost::system::error_code ec_2{};
                    this_socket.shutdown(boost::asio::socket_base::shutdown_both, ec_2);
                    this_socket.close(ec_2);
                }
            } else {
                // TODO
                //StopAccept();
                //return;
            }
            DoAccept();
        });
}

std::shared_ptr<TcpHandler> TcpServer::CreateHandler()
{
    if (m_option.m_max_connection != 0 && m_handlers.size() >= m_option.m_max_connection) {
        m_event->OnAcceptOverflow();
        return nullptr;
    }
    auto handler = std::make_shared<TcpHandler>(std::move(m_socket), *this, ++m_handler_index);
    m_handlers.insert(handler);
    return handler;
}

void TcpServer::Run()
{
    while (m_is_running) {
        try {
            m_io_context.run();
        } catch (const std::exception& e) {
            m_event->OnCatchException(e);
        }
    }
}

void TcpServer::StopHandler(const std::shared_ptr<TcpHandler>& handler)
{
    m_handlers.erase(handler);
}

boost::asio::io_context& TcpServer::GetIOContext()
{
    return m_acceptor.get_io_context();
}

const TcpServer::Option& TcpServer::GetOption() const
{
    return m_option;
}

TcpServerEvent& TcpServer::GetEvent()
{
    return *m_event;
}

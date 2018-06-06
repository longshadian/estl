#pragma once

#include <vector>
#include <string>

class BinaryByteBuffer
{
public:
    BinaryByteBuffer();
    explicit BinaryByteBuffer(std::vector<uint8_t> data);
    explicit BinaryByteBuffer(const std::string& data);
    explicit BinaryByteBuffer(const void* data, size_t len);
    ~BinaryByteBuffer() = default;
    BinaryByteBuffer(const BinaryByteBuffer& rhs) = delete;
    BinaryByteBuffer& operator=(const BinaryByteBuffer& rhs) = delete;
    BinaryByteBuffer(BinaryByteBuffer&& rhs) = delete;
    BinaryByteBuffer& operator=(BinaryByteBuffer&& rhs) = delete;

    size_t                  Capacity() const;
    void                    Resize(size_t len);
    void                    Append(const void* p, size_t len);
    void                    Clear();
    void                    ShrinkToFit();
    void                    MemoryMove();

    size_t                  ReaderIndex() const;
    size_t                  ReadableSize() const;
    const void*             GetReaderPtr() const;
    void                    ReaderPickup(size_t len);

    size_t                  WriterIndex() const;
    size_t                  WritableSize() const;

private:
    std::vector<uint8_t>    m_storage;
    size_t                  m_rpos;
    size_t                  m_wpos;
};

using ByteBuffer = BinaryByteBuffer;

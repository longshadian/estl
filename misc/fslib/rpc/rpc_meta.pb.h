// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: rpc_meta.proto

#ifndef PROTOBUF_rpc_5fmeta_2eproto__INCLUDED
#define PROTOBUF_rpc_5fmeta_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 2005000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 2005000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

namespace fslib {
namespace grpc {

// Internal implementation detail -- do not call these.
void  protobuf_AddDesc_rpc_5fmeta_2eproto();
void protobuf_AssignDesc_rpc_5fmeta_2eproto();
void protobuf_ShutdownFile_rpc_5fmeta_2eproto();

class rpc_meta;

// ===================================================================

class rpc_meta : public ::google::protobuf::Message {
 public:
  rpc_meta();
  virtual ~rpc_meta();

  rpc_meta(const rpc_meta& from);

  inline rpc_meta& operator=(const rpc_meta& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const rpc_meta& default_instance();

  void Swap(rpc_meta* other);

  // implements Message ----------------------------------------------

  rpc_meta* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const rpc_meta& from);
  void MergeFrom(const rpc_meta& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // required int32 id = 1;
  inline bool has_id() const;
  inline void clear_id();
  static const int kIdFieldNumber = 1;
  inline ::google::protobuf::int32 id() const;
  inline void set_id(::google::protobuf::int32 value);

  // required string name = 2;
  inline bool has_name() const;
  inline void clear_name();
  static const int kNameFieldNumber = 2;
  inline const ::std::string& name() const;
  inline void set_name(const ::std::string& value);
  inline void set_name(const char* value);
  inline void set_name(const char* value, size_t size);
  inline ::std::string* mutable_name();
  inline ::std::string* release_name();
  inline void set_allocated_name(::std::string* name);

  // @@protoc_insertion_point(class_scope:fslib.grpc.rpc_meta)
 private:
  inline void set_has_id();
  inline void clear_has_id();
  inline void set_has_name();
  inline void clear_has_name();

  ::google::protobuf::UnknownFieldSet _unknown_fields_;

  ::std::string* name_;
  ::google::protobuf::int32 id_;

  mutable int _cached_size_;
  ::google::protobuf::uint32 _has_bits_[(2 + 31) / 32];

  friend void  protobuf_AddDesc_rpc_5fmeta_2eproto();
  friend void protobuf_AssignDesc_rpc_5fmeta_2eproto();
  friend void protobuf_ShutdownFile_rpc_5fmeta_2eproto();

  void InitAsDefaultInstance();
  static rpc_meta* default_instance_;
};
// ===================================================================


// ===================================================================

// rpc_meta

// required int32 id = 1;
inline bool rpc_meta::has_id() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void rpc_meta::set_has_id() {
  _has_bits_[0] |= 0x00000001u;
}
inline void rpc_meta::clear_has_id() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void rpc_meta::clear_id() {
  id_ = 0;
  clear_has_id();
}
inline ::google::protobuf::int32 rpc_meta::id() const {
  return id_;
}
inline void rpc_meta::set_id(::google::protobuf::int32 value) {
  set_has_id();
  id_ = value;
}

// required string name = 2;
inline bool rpc_meta::has_name() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void rpc_meta::set_has_name() {
  _has_bits_[0] |= 0x00000002u;
}
inline void rpc_meta::clear_has_name() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void rpc_meta::clear_name() {
  if (name_ != &::google::protobuf::internal::kEmptyString) {
    name_->clear();
  }
  clear_has_name();
}
inline const ::std::string& rpc_meta::name() const {
  return *name_;
}
inline void rpc_meta::set_name(const ::std::string& value) {
  set_has_name();
  if (name_ == &::google::protobuf::internal::kEmptyString) {
    name_ = new ::std::string;
  }
  name_->assign(value);
}
inline void rpc_meta::set_name(const char* value) {
  set_has_name();
  if (name_ == &::google::protobuf::internal::kEmptyString) {
    name_ = new ::std::string;
  }
  name_->assign(value);
}
inline void rpc_meta::set_name(const char* value, size_t size) {
  set_has_name();
  if (name_ == &::google::protobuf::internal::kEmptyString) {
    name_ = new ::std::string;
  }
  name_->assign(reinterpret_cast<const char*>(value), size);
}
inline ::std::string* rpc_meta::mutable_name() {
  set_has_name();
  if (name_ == &::google::protobuf::internal::kEmptyString) {
    name_ = new ::std::string;
  }
  return name_;
}
inline ::std::string* rpc_meta::release_name() {
  clear_has_name();
  if (name_ == &::google::protobuf::internal::kEmptyString) {
    return NULL;
  } else {
    ::std::string* temp = name_;
    name_ = const_cast< ::std::string*>(&::google::protobuf::internal::kEmptyString);
    return temp;
  }
}
inline void rpc_meta::set_allocated_name(::std::string* name) {
  if (name_ != &::google::protobuf::internal::kEmptyString) {
    delete name_;
  }
  if (name) {
    set_has_name();
    name_ = name;
  } else {
    clear_has_name();
    name_ = const_cast< ::std::string*>(&::google::protobuf::internal::kEmptyString);
  }
}


// @@protoc_insertion_point(namespace_scope)

}  // namespace grpc
}  // namespace fslib

#ifndef SWIG
namespace google {
namespace protobuf {


}  // namespace google
}  // namespace protobuf
#endif  // SWIG

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_rpc_5fmeta_2eproto__INCLUDED

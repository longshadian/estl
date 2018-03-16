        #include <string>
#include <iostream>
#include <vector>
#include <memory>

#include "msg_gbmj.pb.h"

#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/type_resolver_util.h>

const std::string URL = "type.googleapis.com";

std::string toString(const std::vector<uint8_t>& data)
{
    return std::string(data.begin(), data.end());
}

std::string testJson(const ::pt::zj::xx& req)
{
    std::string s{};
    google::protobuf::util::JsonPrintOptions printOpt{};
    printOpt.always_print_primitive_fields = true;
    printOpt.add_whitespace = true;
    if (google::protobuf::util::MessageToJsonString(req, &s, printOpt).ok()) {
        std::cout << s << "\n";
    } else {
        std::cout << s << "\n";
    }
    return s;
}

std::string testBinary(const ::pt::zj::xx& req)
{
    std::vector<uint8_t> binary_data{};
    binary_data.resize(req.ByteSize());
    req.SerializeToArray(binary_data.data(), binary_data.size());

    google::protobuf::util::TypeResolver* resolver =
        google::protobuf::util::NewTypeResolverForDescriptorPool(
            URL, google::protobuf::DescriptorPool::generated_pool());
    if (!resolver) {
        std::cout << "resolver error\n";
        return "";
    }

    std::string json_str;
    auto ret = google::protobuf::util::BinaryToJsonString(resolver
        , URL + "/pt.zj.xx", toString(binary_data), &json_str);
    //std::cout << ret.ok() << "\n";
    std::cout << json_str << "\n";
    return json_str;
}

int main()
{
    ::pt::zj::xx req{};
    req.set_v_1(1);
    req.set_v2(2);
    req.set_str("111111");
    for (int i = 0; i != 4; ++i) {
        auto* user = req.add_user();
        user->set_s(::pt::zj::obj_user::S_SUCCESS_1);
        user->add_arr(i + 1);
        user->add_arr(i + 2);
    }
    req.mutable_test_1()->set_val_string("aaa");
    req.set_username("xxx");
    auto s1 = testJson(req);

    ::pt::zj::xx req_ex{};
    auto ret = ::google::protobuf::util::JsonStringToMessage(s1, &req_ex);
    std::cout << ret.ok() << "\n";
    std::cout << req_ex.username() << "\n";

    /*
    auto s2 = testBinary(req);
    std::cout << (s1 == s2) << "\n";
    */
    return 0;
}

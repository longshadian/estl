//
//  upload.cc
//
//  Copyright (c) 2019 Yuji Hirose. All rights reserved.
//  MIT License
//

#include <fstream>
#include <httplib.h>
#include <iostream>
using namespace httplib;
using namespace std;

const char *html = R"(
<form id="formElem">
  <input type="file" name="image_file" accept="image/*">
  <input type="file" name="text_file" accept="text/*">
  <input type="submit">
</form>
<script>
  formElem.onsubmit = async (e) => {
    e.preventDefault();
    let res = await fetch('/post', {
      method: 'POST',
      body: new FormData(formElem)
    });
    console.log(await res.text());
  };
</script>
)";

int main(void) {
  Server svr;

  svr.Get("/", [](const Request & /*req*/, Response &res) {
    res.set_content(html, "text/html");
  });

#if 0
  svr.Post("/post", [](const Request &req, Response &res) {
    auto image_file = req.get_file_value("image_file");
    auto text_file = req.get_file_value("text_file");

    cout << "image file length: " << image_file.content.length() << endl
         << "image file name: " << image_file.filename << endl
         << "text file length: " << text_file.content.length() << endl
         << "text file name: " << text_file.filename << endl;

    {
      ofstream ofs(image_file.filename, ios::binary);
      ofs << image_file.content;
    }
    {
      ofstream ofs(text_file.filename);
      ofs << text_file.content;
    }

    res.set_content("done", "text/plain");
  });
#else
  svr.Post("/post", [](const Request &req, Response &res) {

    for (const auto& file : req.files) {
        const auto& file_key = file.first;
        const auto& file_value = file.second;

        cout << "file_key: " << file_key << " content_length: " << file_value.content.length() 
            << " filename: " << file_value.filename  << " content_type: " << file_value.content_type
            << std::endl;
          ofstream ofs(file_value.filename, ios::binary);
          ofs << file_value.content;
    }
    res.set_content("done", "text/plain");
  });

#endif

  svr.listen("0.0.0.0", 10086);
}

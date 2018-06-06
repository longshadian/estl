生成私钥
openssl genrsa -out private.key 2048

用私钥生成公钥
openssl rsa -in private.key -pubout -out public.key

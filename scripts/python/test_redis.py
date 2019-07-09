import redis


file = '''

--[[
    删除timestamp之前的record数据
    KEYS[1]    timestamp  时间戳
	return 	删除的数量
--]]

redis.log(redis.LOG_NOTICE, "gp_remove_record_cached")
local timestamp = tonumber(KEYS[1])
local cnt = 0

local cursor = 0
local count = 100
while true do
    local result = redis.pcall("HSCAN", "record_all", cursor, "count", count);
    local next_cursor = tonumber(result[1])
    local list = result[2]
    if list ~= nil then
        for i=1, #list, 2 do 
            local room_uuid = list[i]
            local create_tm = tonumber(list[i + 1])
            --redis.log(redis.LOG_NOTICE, string.format("xxx %d %s %d", next_cursor, room_uuid, create_tm))
            --if create_tm <= timestamp then
                cnt = cnt + 1
                --redis.pcall("HDEL", "gprecord_tm", room_uuid)
                --redis.pcall("HDEL", "gprecord", room_uuid)
                ----redis.log(redis.LOG_NOTICE, string.format("user_remove_user succ user_id:%s", user_id))
            --end
        end
    end
    
    redis.log(redis.LOG_NOTICE, string.format("xxx %d", next_cursor))
    
    ----遍历结束
    if next_cursor == 0 then
        break
    end
    cursor = next_cursor
    
end

return cnt

'''


def test_eval():
    r = redis.StrictRedis(host="192.168.0.242", port=21115)
    ret = r.eval(file, 1, 1524896940)
    print(ret)


def main():
    test_eval()


if __name__ == '__main__':
    main()

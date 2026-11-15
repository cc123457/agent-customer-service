"""天气 MCP 服务：通过 MCP 协议暴露真实天气查询工具。

数据源 Open-Meteo（免费、免注册、免 key）。
独立进程运行（streamable-http，端口 8001）：
    python -m mcp_servers.weather_server
"""
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather", host="127.0.0.1", port=8001)

# Open-Meteo 天气代码 → 中文描述
_WEATHER_CODE = {
    0: "晴", 1: "大致晴朗", 2: "多云", 3: "阴", 45: "雾", 48: "雾凇",
    51: "毛毛雨", 53: "小雨", 55: "中雨", 61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪", 80: "阵雨", 81: "强阵雨", 82: "暴雨",
    95: "雷暴", 96: "雷暴伴冰雹", 99: "强雷暴伴冰雹",
}


@mcp.tool()
async def get_weather(city: str) -> str:
    """根据城市名称查询当前实时天气。

    Args:
        city: 城市名称，支持中文或英文，如「北京」「上海」「Tokyo」。

    Returns:
        一段中文天气描述（含天气状况、气温、湿度、风速）。
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1) 地理编码：城市名 → 经纬度
        geo = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "zh", "format": "json"},
        )
        geo.raise_for_status()
        results = geo.json().get("results")
        if not results:
            return f"未找到城市「{city}」，请确认名称是否正确。"
        loc = results[0]
        lat, lon = loc["latitude"], loc["longitude"]
        name = loc.get("name", city)
        country = loc.get("country", "")
        admin = loc.get("admin1", "")

        # 2) 查询该坐标的当前天气
        wx = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
        )
        wx.raise_for_status()
        current = wx.json().get("current", {})
        code = current.get("weather_code")
        desc = _WEATHER_CODE.get(code, f"未知天气({code})")
        place = "".join(p for p in [country, admin, name] if p)
        return (
            f"{place} 当前天气：{desc}，气温 {current.get('temperature_2m')}°C，"
            f"相对湿度 {current.get('relative_humidity_2m')}%，"
            f"风速 {current.get('wind_speed_10m')} km/h。（数据来源：Open-Meteo）"
        )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")

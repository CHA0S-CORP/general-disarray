"""
Weather Tool Plugin
===================
Get current weather from a Tempest weather station.

Requires configuration:
- TEMPEST_STATION_ID: Your Tempest station ID
- TEMPEST_API_TOKEN: Your Tempest API token

Usage in conversation:
User: "What's the weather like?"
LLM: [TOOL:WEATHER]
"""

import logging
import httpx
from datetime import datetime
from typing import Any, Dict, Optional

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import log_event

logger = logging.getLogger(__name__)


class WeatherTool(BaseTool):
    """Get current weather from Tempest weather station."""
    
    name = "WEATHER"
    description = "Get current weather conditions from the local weather station"
    enabled = True
    
    parameters = {}  # No parameters needed - uses configured station
    
    def __init__(self, assistant):
        super().__init__(assistant)
        # Check if weather is configured
        if self.config:
            if not self.config.tempest_station_id or not self.config.tempest_api_token:
                self.enabled = False
                logger.info("Weather tool disabled - Tempest API not configured")
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        station_id = self.config.tempest_station_id
        api_token = self.config.tempest_api_token
        
        if not station_id or not api_token:
            return ToolResult(
                status=ToolStatus.FAILED,
                message="Weather station not configured"
            )
        
        try:
            url = f"https://swd.weatherflow.com/swd/rest/observations/station/{station_id}"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params={"token": api_token})
                
                if response.status_code != 200:
                    logger.error(f"Weather API error: {response.status_code}")
                    return ToolResult(
                        status=ToolStatus.FAILED,
                        message="Couldn't reach the weather station"
                    )
                
                data = response.json()
            
            # Parse response
            station_name = data.get("station_name", "the weather station")
            timezone = data.get("timezone")
            obs_list = data.get("obs", [])
            
            if not obs_list:
                return ToolResult(
                    status=ToolStatus.FAILED,
                    message="No weather data available"
                )
            
            obs = obs_list[0]
            
            # Build natural weather summary
            summary = self._build_summary(obs, station_name, timezone)
            
            # Log and return
            log_event(logger, logging.INFO, f"Weather: {summary}", event="weather_fetch")
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                message=summary,
                data=self._extract_data(obs)
            )
            
        except httpx.TimeoutException:
            return ToolResult(status=ToolStatus.FAILED, message="Weather station timed out")
        except Exception as e:
            logger.error(f"Weather error: {e}", exc_info=True)
            return ToolResult(status=ToolStatus.FAILED, message="Error getting weather data")
    
    def _build_summary(self, obs: dict, station_name: str, timezone: str = None) -> str:
        """Build a natural, conversational weather summary."""
        parts = []
        
        # Format observation time
        timestamp = obs.get("timestamp")
        time_str = ""
        if timestamp:
            try:
                from datetime import datetime
                import pytz
                
                obs_time = datetime.fromtimestamp(timestamp)
                if timezone:
                    try:
                        tz = pytz.timezone(timezone)
                        obs_time = datetime.fromtimestamp(timestamp, tz)
                    except:
                        pass
                
                # Format as "3:45 PM"
                time_str = obs_time.strftime("%-I:%M %p").lower()
            except:
                pass
        
        # Opening with station and time
        if time_str:
            parts.append(f"At {station_name}, as of {time_str}")
        else:
            parts.append(f"At {station_name}")
        
        # Temperature (convert C to F)
        temp_c = obs.get("air_temperature")
        feels_like_c = obs.get("feels_like")
        
        if temp_c is not None:
            temp_f = round(temp_c * 9/5 + 32)
            feels_like_f = round(feels_like_c * 9/5 + 32) if feels_like_c is not None else temp_f
            
            # Determine conditions based on solar radiation and time
            solar = obs.get("solar_radiation", 0)
            brightness = obs.get("brightness", 0)
            
            if solar == 0 and brightness == 0:
                condition = ""
            elif solar > 800:
                condition = "sunny and "
            elif solar > 400:
                condition = "partly cloudy and "
            elif solar > 0:
                condition = "overcast and "
            else:
                condition = ""
            
            if abs(temp_f - feels_like_f) <= 2:
                parts.append(f"it's {condition}{temp_f} degrees")
            else:
                parts.append(f"it's {condition}{temp_f} degrees, feels like {feels_like_f}")
        
        # Humidity and fog potential
        humidity = obs.get("relative_humidity")
        dew_point_c = obs.get("dew_point")
        
        if humidity is not None:
            if humidity >= 95 and temp_c is not None and dew_point_c is not None:
                if abs(temp_c - dew_point_c) < 1:
                    parts.append("with foggy conditions")
                else:
                    parts.append(f"with {round(humidity)}% humidity")
            elif humidity >= 85:
                parts.append(f"and very humid at {round(humidity)}%")
            elif humidity <= 30:
                parts.append("and quite dry")
        
        # Wind
        wind_avg = obs.get("wind_avg", 0)
        wind_gust = obs.get("wind_gust", 0)
        wind_dir = obs.get("wind_direction", 0)
        
        if wind_avg is not None:
            wind_mph = round(wind_avg * 2.237)
            gust_mph = round(wind_gust * 2.237) if wind_gust else 0
            
            if wind_mph == 0:
                parts.append("Wind is calm")
            else:
                direction = self._wind_direction(wind_dir)
                wind_str = f"Wind from the {direction} at {wind_mph} mph"
                
                if gust_mph > wind_mph + 5:
                    wind_str += f", gusting to {gust_mph}"
                
                if wind_mph >= 20:
                    wind_str += ". It's quite windy"
                elif wind_mph >= 15:
                    wind_str += ". A bit breezy"
                    
                parts.append(wind_str)
        
        # Precipitation
        precip_now = obs.get("precip", 0)
        precip_hour = obs.get("precip_accum_last_1hr", 0)
        precip_today = obs.get("precip_accum_local_day", 0)
        precip_yesterday = obs.get("precip_accum_local_yesterday_final", 0)
        
        if precip_now and precip_now > 0:
            intensity = "lightly" if precip_now < 0.5 else "moderately" if precip_now < 2 else "heavily"
            parts.append(f"It's {intensity} raining")
            if precip_today > 0:
                parts.append(f"with {round(precip_today * 0.0394, 2)} inches so far today")
        elif precip_hour and precip_hour > 0:
            parts.append(f"Rain stopped recently, {round(precip_hour * 0.0394, 2)} inches in the past hour")
        elif precip_today and precip_today > 0:
            parts.append(f"{round(precip_today * 0.0394, 2)} inches of rain today")
        elif precip_yesterday and precip_yesterday > 0:
            parts.append(f"Yesterday saw {round(precip_yesterday * 0.0394, 2)} inches of rain")
        
        # Lightning
        lightning_1hr = obs.get("lightning_strike_count_last_1hr", 0)
        lightning_3hr = obs.get("lightning_strike_count_last_3hr", 0)
        lightning_dist = obs.get("lightning_strike_last_distance")
        
        if lightning_1hr and lightning_1hr > 0:
            dist_str = f"{round(lightning_dist * 0.621)} miles away" if lightning_dist else "nearby"
            parts.append(f"Lightning detected! {lightning_1hr} strikes in the past hour, last one {dist_str}")
        elif lightning_3hr and lightning_3hr > 0:
            parts.append(f"There were {lightning_3hr} lightning strikes in the past 3 hours")
        
        # UV warning (daytime only)
        uv = obs.get("uv", 0)
        if uv and uv >= 6:
            if uv >= 8:
                parts.append(f"UV index is very high at {round(uv)}, wear sunscreen")
            else:
                parts.append(f"UV index is {round(uv)}, moderate sun protection advised")
        
        # Pressure trend
        pressure = obs.get("barometric_pressure")
        trend = obs.get("pressure_trend")
        
        if trend and trend != "steady" and pressure:
            if trend == "rising":
                parts.append("Barometric pressure is rising, weather may be improving")
            elif trend == "falling":
                parts.append("Barometric pressure is falling, weather may be changing")
        
        # Combine into natural speech
        if not parts:
            return f"Current conditions at {station_name} are unavailable."
        
        # Join with proper punctuation
        result = parts[0]
        for i, part in enumerate(parts[1:], 1):
            if part[0].isupper():
                result += ". " + part
            else:
                result += ", " + part
        
        return result + "."
    
    def _wind_direction(self, degrees: float) -> str:
        """Convert degrees to cardinal direction."""
        if degrees is None:
            return "unknown"
        directions = [
            "north", "north-northeast", "northeast", "east-northeast",
            "east", "east-southeast", "southeast", "south-southeast",
            "south", "south-southwest", "southwest", "west-southwest",
            "west", "west-northwest", "northwest", "north-northwest"
        ]
        idx = int((degrees + 11.25) / 22.5) % 16
        return directions[idx]
    
    def _extract_data(self, obs: dict) -> dict:
        """Extract structured data for the response."""
        temp_c = obs.get("air_temperature")
        feels_c = obs.get("feels_like")
        wind = obs.get("wind_avg")
        gust = obs.get("wind_gust")
        
        return {
            "temp_f": round(temp_c * 9/5 + 32) if temp_c is not None else None,
            "feels_like_f": round(feels_c * 9/5 + 32) if feels_c is not None else None,
            "humidity": obs.get("relative_humidity"),
            "wind_mph": round(wind * 2.237) if wind is not None else None,
            "wind_gust_mph": round(gust * 2.237) if gust is not None else None,
            "wind_direction": obs.get("wind_direction"),
            "precip_today_in": round(obs.get("precip_accum_local_day", 0) * 0.0394, 2),
            "uv": obs.get("uv"),
            "pressure_mb": obs.get("barometric_pressure"),
            "pressure_trend": obs.get("pressure_trend"),
            "lightning_1hr": obs.get("lightning_strike_count_last_1hr"),
            "dew_point_f": round(obs.get("dew_point", 0) * 9/5 + 32) if obs.get("dew_point") else None,
        }
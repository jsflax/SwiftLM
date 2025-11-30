#!/usr/bin/env python3
"""
MapKit POI Search using PyObjC

Provides access to Apple Maps POI search for generating realistic training data
for skeleton planning and activity planning stages.

Requires: pyobjc-framework-CoreLocation, pyobjc-framework-MapKit

Usage:
    from mapkit_poi import search_pois, search_pois_in_region

    # Search for restaurants near a location
    results = search_pois("restaurants", latitude=35.6762, longitude=139.6503)

    # Search in a region (for broader area search)
    results = search_pois_in_region("surf spots", "Santa Teresa, Costa Rica", radius_km=50)
"""

import objc
from typing import Optional
import time
import threading
import math

# Import MapKit and CoreLocation frameworks
try:
    from MapKit import (
        MKLocalSearch,
        MKLocalSearchRequest,
        MKCoordinateRegion,
        MKCoordinateSpan,
        MKPointOfInterestCategory,
        MKPointOfInterestFilter,
    )
    from CoreLocation import CLLocationCoordinate2D, CLGeocoder, CLLocation
    from Foundation import NSRunLoop, NSDate
except ImportError as e:
    print(f"Error importing MapKit/CoreLocation: {e}")
    print("Install with: pip install pyobjc-framework-MapKit pyobjc-framework-CoreLocation")
    raise

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.

    Args:
        lat1, lon1: First point coordinates in degrees
        lat2, lon2: Second point coordinates in degrees

    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers

    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(delta_lat / 2) ** 2 + \
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# POI category mappings for trip planning
# Maps our activity types to MKPointOfInterestCategory constants
POI_CATEGORIES = {
    # Food & Drink
    "restaurant": "MKPointOfInterestCategoryRestaurant",
    "cafe": "MKPointOfInterestCategoryCafe",
    "food": "MKPointOfInterestCategoryFoodMarket",
    "bakery": "MKPointOfInterestCategoryBakery",
    "brewery": "MKPointOfInterestCategoryBrewery",
    "winery": "MKPointOfInterestCategoryWinery",
    "nightlife": "MKPointOfInterestCategoryNightlife",

    # Culture & Entertainment
    "museum": "MKPointOfInterestCategoryMuseum",
    "theater": "MKPointOfInterestCategoryTheater",
    "movie": "MKPointOfInterestCategoryMovieTheater",
    "amusement": "MKPointOfInterestCategoryAmusementPark",
    "zoo": "MKPointOfInterestCategoryZoo",
    "aquarium": "MKPointOfInterestCategoryAquarium",

    # Outdoors & Recreation
    "park": "MKPointOfInterestCategoryPark",
    "beach": "MKPointOfInterestCategoryBeach",
    "marina": "MKPointOfInterestCategoryMarina",
    "campground": "MKPointOfInterestCategoryCampground",
    "nationalPark": "MKPointOfInterestCategoryNationalPark",

    # Sports & Fitness
    "fitness": "MKPointOfInterestCategoryFitnessCenter",
    "stadium": "MKPointOfInterestCategoryStadium",
    "golf": "MKPointOfInterestCategoryGolfCourse",

    # Services
    "spa": "MKPointOfInterestCategorySpa",
    "hotel": "MKPointOfInterestCategoryHotel",
    "store": "MKPointOfInterestCategoryStore",

    # Transportation
    "airport": "MKPointOfInterestCategoryAirport",
    "publicTransport": "MKPointOfInterestCategoryPublicTransport",
}

# Activity search terms for trip planning stages
ACTIVITY_SEARCH_TERMS = {
    "surfing": ["surf shop", "surf school", "beach", "surf rental"],
    "hiking": ["hiking trail", "national park", "nature reserve", "mountain"],
    "diving": ["dive shop", "dive center", "scuba", "snorkeling"],
    "food": ["restaurant", "local food", "street food", "food market", "cafe"],
    "culture": ["museum", "art gallery", "temple", "shrine", "historic site"],
    "nightlife": ["bar", "club", "nightlife", "live music", "cocktail bar"],
    "wellness": ["spa", "yoga studio", "wellness center", "hot springs", "onsen"],
    "shopping": ["shopping mall", "market", "boutique", "vintage shop"],
    "photography": ["viewpoint", "scenic", "landmark", "observation deck"],
    "wildlife": ["zoo", "aquarium", "wildlife sanctuary", "nature reserve"],
}


class MapKitPOISearch:
    """Wrapper for MapKit local search functionality."""

    def __init__(self):
        self.geocoder = CLGeocoder.alloc().init()
        self._pending_results = None
        self._search_complete = False
        self._geocode_complete = False

    def geocode_location(self, address: str, timeout: float = 10.0) -> Optional[tuple]:
        """
        Geocode an address to coordinates.

        Args:
            address: Location string (e.g., "Tokyo, Japan" or "Santa Teresa, Costa Rica")
            timeout: Maximum time to wait for geocoding

        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        self._geocode_complete = False
        self._geocode_result = None

        def handler(placemarks, error):
            if error:
                print(f"Geocoding error: {error}")
                self._geocode_result = None
            elif placemarks and len(placemarks) > 0:
                location = placemarks[0].location()
                if location:
                    coord = location.coordinate()
                    self._geocode_result = (coord.latitude, coord.longitude)
                else:
                    self._geocode_result = None
            else:
                self._geocode_result = None
            self._geocode_complete = True

        self.geocoder.geocodeAddressString_completionHandler_(address, handler)

        # Run the run loop to process async callbacks
        run_loop = NSRunLoop.currentRunLoop()
        start_time = time.time()
        while not self._geocode_complete:
            # Process events for a short interval
            run_loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))
            if time.time() - start_time > timeout:
                print(f"Geocoding timeout for: {address}")
                return None

        return self._geocode_result

    def search(
        self,
        query: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        location_name: Optional[str] = None,
        radius_km: float = 10.0,
        limit: int = 20,
        timeout: float = 15.0,
        filter_by_distance: bool = True,
    ) -> list:
        """
        Search for POIs using MapKit local search.

        Args:
            query: Search query (e.g., "restaurants", "surf spots", "temples")
            latitude: Center latitude (optional if location_name provided)
            longitude: Center longitude (optional if location_name provided)
            location_name: Location string to geocode (e.g., "Kyoto, Japan")
            radius_km: Search radius in kilometers
            limit: Maximum number of results
            timeout: Maximum time to wait for search
            filter_by_distance: If True, exclude results outside radius_km (default: True)

        Returns:
            List of POI dictionaries with name, category, coordinates, address, distance_km, etc.
        """
        # Geocode if needed
        if latitude is None or longitude is None:
            if location_name:
                coords = self.geocode_location(location_name)
                if coords:
                    latitude, longitude = coords
                else:
                    print(f"Could not geocode: {location_name}")
                    return []
            else:
                raise ValueError("Must provide either (latitude, longitude) or location_name")

        # Store center for distance calculation
        center_lat, center_lon = latitude, longitude

        # Create search request
        request = MKLocalSearchRequest.alloc().init()
        request.setNaturalLanguageQuery_(query)

        # Set search region
        center = CLLocationCoordinate2D(latitude, longitude)
        # Convert km to degrees (approximate: 1 degree ≈ 111 km)
        span_degrees = radius_km / 111.0
        span = MKCoordinateSpan(span_degrees, span_degrees)
        region = MKCoordinateRegion(center, span)
        request.setRegion_(region)

        # Create and start search
        search = MKLocalSearch.alloc().initWithRequest_(request)

        self._search_complete = False
        self._search_results = []

        def search_handler(response, error):
            if error:
                print(f"Search error: {error}")
                self._search_results = []
            elif response:
                map_items = response.mapItems()
                results = []
                for item in map_items:
                    placemark = item.placemark()
                    coord = placemark.coordinate()

                    # Calculate distance from search center
                    distance_km = haversine_distance(
                        center_lat, center_lon,
                        coord.latitude, coord.longitude
                    )

                    # Skip if outside search radius (when filtering enabled)
                    if filter_by_distance and distance_km > radius_km:
                        continue

                    result = {
                        "name": item.name() or "Unknown",
                        "latitude": coord.latitude,
                        "longitude": coord.longitude,
                        "distance_km": round(distance_km, 2),
                        "address": self._format_address(placemark),
                        "phone": item.phoneNumber() or None,
                        "url": str(item.url()) if item.url() else None,
                        "category": self._get_category(item),
                    }
                    results.append(result)

                    if len(results) >= limit:
                        break

                # Sort by distance
                results.sort(key=lambda x: x["distance_km"])
                self._search_results = results
            else:
                self._search_results = []
            self._search_complete = True

        search.startWithCompletionHandler_(search_handler)

        # Run the run loop to process async callbacks
        run_loop = NSRunLoop.currentRunLoop()
        start_time = time.time()
        while not self._search_complete:
            # Process events for a short interval
            run_loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))
            if time.time() - start_time > timeout:
                print(f"Search timeout for: {query}")
                return []

        return self._search_results

    def _format_address(self, placemark) -> str:
        """Format a placemark into a readable address."""
        parts = []
        if placemark.subThoroughfare():
            parts.append(placemark.subThoroughfare())
        if placemark.thoroughfare():
            parts.append(placemark.thoroughfare())
        if placemark.locality():
            parts.append(placemark.locality())
        if placemark.administrativeArea():
            parts.append(placemark.administrativeArea())
        if placemark.country():
            parts.append(placemark.country())
        return ", ".join(filter(None, parts))

    def _get_category(self, map_item) -> Optional[str]:
        """Extract category from a map item if available."""
        # MKMapItem category is available on newer systems
        try:
            category = map_item.pointOfInterestCategory()
            if category:
                # Remove the MKPointOfInterestCategory prefix
                return str(category).replace("MKPointOfInterestCategory", "").lower()
        except:
            pass
        return None

    def search_activities(
        self,
        activity_type: str,
        location_name: str,
        radius_km: float = 20.0,
        limit_per_term: int = 10,
    ) -> list:
        """
        Search for activities of a given type in a location.
        Uses multiple search terms to get comprehensive results.

        Args:
            activity_type: Type of activity (e.g., "surfing", "food", "culture")
            location_name: Location string (e.g., "Santa Teresa, Costa Rica")
            radius_km: Search radius
            limit_per_term: Max results per search term

        Returns:
            List of unique POIs
        """
        search_terms = ACTIVITY_SEARCH_TERMS.get(activity_type, [activity_type])

        # Get coordinates once
        coords = self.geocode_location(location_name)
        if not coords:
            print(f"Could not geocode: {location_name}")
            return []

        latitude, longitude = coords

        all_results = []
        seen_names = set()

        for term in search_terms:
            results = self.search(
                query=term,
                latitude=latitude,
                longitude=longitude,
                radius_km=radius_km,
                limit=limit_per_term,
            )

            for r in results:
                # Deduplicate by name
                if r["name"] not in seen_names:
                    seen_names.add(r["name"])
                    r["search_term"] = term
                    r["activity_type"] = activity_type
                    all_results.append(r)

            # Small delay between searches to be nice to the API
            time.sleep(0.5)

        return all_results


# Convenience functions
_searcher = None

def get_searcher() -> MapKitPOISearch:
    """Get or create the global searcher instance."""
    global _searcher
    if _searcher is None:
        _searcher = MapKitPOISearch()
    return _searcher


def search_pois(
    query: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    location_name: Optional[str] = None,
    radius_km: float = 10.0,
    limit: int = 20,
    filter_by_distance: bool = True,
) -> list:
    """
    Search for POIs near a location.

    Examples:
        # Search by coordinates
        results = search_pois("ramen", latitude=35.6762, longitude=139.6503)

        # Search by location name
        results = search_pois("surf shop", location_name="Santa Teresa, Costa Rica")

        # Search without distance filtering
        results = search_pois("park", location_name="NYC", filter_by_distance=False)
    """
    return get_searcher().search(
        query=query,
        latitude=latitude,
        longitude=longitude,
        location_name=location_name,
        radius_km=radius_km,
        limit=limit,
        filter_by_distance=filter_by_distance,
    )


def search_activities(
    activity_type: str,
    location_name: str,
    radius_km: float = 20.0,
) -> list:
    """
    Search for activities of a given type.

    Examples:
        # Find surfing spots
        results = search_activities("surfing", "Santa Teresa, Costa Rica")

        # Find food options
        results = search_activities("food", "Kyoto, Japan")
    """
    return get_searcher().search_activities(
        activity_type=activity_type,
        location_name=location_name,
        radius_km=radius_km,
    )


def geocode(address: str) -> Optional[tuple]:
    """
    Geocode an address to coordinates.

    Example:
        coords = geocode("Tokyo Tower, Japan")
        print(f"Lat: {coords[0]}, Lon: {coords[1]}")
    """
    return get_searcher().geocode_location(address)


# Test/demo code
if __name__ == "__main__":
    print("MapKit POI Search Demo")
    print("=" * 50)

    # Test geocoding
    print("\n1. Testing geocoding...")
    coords = geocode("Santa Teresa, Costa Rica")
    if coords:
        print(f"   Santa Teresa: {coords[0]:.4f}, {coords[1]:.4f}")
    else:
        print("   Geocoding failed")

    # Test POI search
    print("\n2. Testing POI search for 'surf shop'...")
    results = search_pois("surf shop", location_name="Santa Teresa, Costa Rica", limit=5)
    for r in results:
        print(f"   - {r['name']}")
        print(f"     {r['address']}")

    # Test activity search
    print("\n3. Testing activity search for 'surfing'...")
    results = search_activities("surfing", "Santa Teresa, Costa Rica")
    print(f"   Found {len(results)} results")
    for r in results[:5]:
        print(f"   - {r['name']} ({r.get('search_term', 'unknown')})")

    # Test another location
    print("\n4. Testing food search in Tokyo...")
    results = search_pois("ramen", location_name="Shibuya, Tokyo, Japan", limit=5)
    for r in results:
        print(f"   - {r['name']}")
        print(f"     {r['address']}")

    print("\n" + "=" * 50)
    print("Demo complete!")

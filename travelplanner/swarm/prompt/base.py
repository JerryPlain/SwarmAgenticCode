TASK_MINI = '''Based on the provided reference information and query, please give me a detailed travel plan. The plan should include the following items:
- Transportation: Choose 'Flight', 'Self-driving' or 'Taxi' for intercity travel throughout the entire trip. 
- Accommodation: Plan the number of nights in each city. For each city, recommend suitable accommodations that align with the travel schedule —or, if none exactly meet all constraints, the second suitable alternative—and specify how many rooms are needed to comfortably accommodate all travelers.
- Restaurants: Provide exactly only one restaurant per meal for each day.
- Attraction: Attraction recommendations for each day.
'''

TASK_EXAMPLE = '''
***** Example *****
Query: Could you create a challenging travel plan from Roanoke to Illinois spanning a week, from March 8th to March 14th, 2022, with a budget of $30,200? The preference is for an entire room, and we would not be taking any flights. In terms of cuisine, we are interested in sampling some Italian and Chinese food. 

Travel Plan: 
Day 1: 
Current City: from Ithaca to Charlotte 
Transportation: Flight Number: F3633413 , from Ithaca to Charlotte , Departure Time: 05:38, Arrival Time: 07:46 
Breakfast: Nagaland 's Kitchen , Charlotte 
Attraction: The Charlotte Museum of History , Charlotte 
Lunch: Cafe Maple Street , Charlotte 
Dinner: Bombay Vada Pav , Charlotte 
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte 

Day 2: 
Current City: Charlotte 
Transportation: -
Breakfast: Olive Tree Cafe , Charlotte 
Attraction: The Mint Museum , Charlotte;Romare Bearden Park , Charlotte. 
Lunch: Birbal Ji Dhaba , Charlotte 
Dinner: Pind Balluchi , Charlotte 
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 3: 
Current City: Charlotte 
Transportation: Flight Number: F3786167 , from Charlotte to Ithaca , Departure Time: 21:42, Arrival Time: 23:26
Breakfast: Subway , Charlotte 
Attraction: Books Monument , Charlotte. 
Lunch: Olive Tree Cafe , Charlotte 
Dinner: Kylin Skybar , Charlotte 
Accommodation: -
***** Example Ends *****
'''

TASK_NOTICE = '''
- For each item in the trip plan, use following **format**:
    - day: Use the numerical value representing the sequence of the day within the travel plan. For instance, '1' for the first day, '2' for the second day, and so on.
    - current city: When there is a change in location, use format "from [City A] to [City B]" to denote the transition. If remaining in the same city, simply use the city's name (e.g., "City A").
    - transportation: For flights, include the details in the format "Flight Number: XXX, from [City A] to [City B]". For self-driving or taxi travel, use "self-driving, from [City A] to [City B]" or "taxi, from [City A] to [City B]". If there is no travel between cities on that day, use "-".
    - breakfast, lunch and dinner: Use "Name, City" to specify the chosen restaurant and its location.
    - attraction: List attraction with its name and located city as 'Name, City;' and use a ';' to separate different attractions, e.g. "Peoria Historical Society, Peoria; Peoria Holocaust Memorial, Peoria;".
    - accommodation: Use "Name, City" to specify the chosen accommodation and its location.
- All details should align with commonsense. 
- You do not need to plan after returning to the departure city.
- All items in the plan should be derived directly from the provided reference information, without making any additional inferences or assumptions. 
- Prices of the Accommodations are the cost per night, not total nights.
- For all days except the last day, every itinerary must explicitly include accommodation arrangements, ensuring no omission occurs, even for consecutive nights in the same accommodation.
- Please retain the original formatting of names exactly as provided in the reference information. Do Not Correct Capitalization or Punctuation in Names.
'''
# - The maximum occupancy listed for an accommodation refers to the capacity per room or unit.

FUNCTION_DESCRIPTION = """
The function is responsible for orchestrating the creation of a comprehensive travel plan by coordinating with various specialists. 
The function signature should be 'def forward(team):'.
The function returns a detailed travel plan.
"""


TASK_OUTPUT_SCHEMA = {
    "title": "Trip_Plan",
    "description": "Detailed trip plan.",
    "type": "object",
    "properties": {
        "travel_plan": {
            "type": "array",
            "description": "Trip plan from the first day to the last day.",
            "items": {
                "type": "object",
                "properties": {
                    "days": {
                         "type": "integer",
                         "description": "Indicates the specific day in the itinerary. Enter the numerical value representing the sequence of the day within the travel plan. For instance, '1' for the first day, '2' for the second day, and so on.",
                    },
                    "current_city": {
                         "type": "string",
                         "description": '''Indicates the city where the traveler is currently located. When there is a change in location, use "from [City A] to [City B]" to denote the transition. If remaining in the same city, simply use the city's name (e.g., "City A").''',
                    },
                    "transportation": {
                        "type": "string",
                        "description": '''Transportations from one city to another. The available transportations are 'Flight', 'Self -driving' and 'Taxi'. For flights, include the details in the format "Flight Number: XXX, from [City A] to [City B]". For self-driving or taxi travel, use "self-driving, from [City A] to [City B]" or "taxi, from [City A] to [City B]". If there is no travel between cities on that day, use "-".''',
                    },
                    "breakfast": {
                        "type": "string",
                        "description": '''Restaurant to have breakfast. Use "Name, City" to specify the chosen restaurant and its location.''',
                    },
                    "attraction": {
                        "type": "string",
                        "description": '''Information about attractions visited. List attraction with its name and located city as 'Name, City;' and use a ';' to separate different attractions, e.g. "Peoria Historical Society, Peoria; Peoria Holocaust Memorial, Peoria;". If no attraction is planned, use "-".''',
                    },
                    "lunch": {
                        "type": "string",
                        "description": '''Restaurant to have lunch. Use "Name, City" to specify the chosen restaurant and its location.''',
                    },
                    "dinner": {
                        "type": "string",
                        "description": '''Restaurant to have dinner. Use "Name, City" to specify the chosen restaurant and its location.''',
                    },
                    "accommodation": {
                        "type": "string",
                        "description": '''Details about accommodation. Use "Name, City" to specify the chosen accommodation and its location. If an accommodation is not planned, use "-".''',
                    },
                },
                "required": ["days", "current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation"],
            }
        },
    },
    "required": ["travel_plan"],
}
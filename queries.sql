select venue, city, count(*) as event_count from events 
where category in ('Category A', 'Category B', 'Category C') and country = 'Italy' and date_iso > '2023-01-01'
group by venue order by event_count desc;

select city, count(*) as event_count from events 
where category in ('Category A', 'Category B', 'Category C') and country = 'Italy' and date_iso > '2023-01-01'
group by city order by event_count desc;

select ev.city,
       ev.venue,
       count(*) as event_count,
       ev.arena_capacity,
       ev.arena_outside_inside,
       ev.arena_type
from events ev
where ev.category in ('Category A', 'Category B', 'Category C')
  and ev.city in ('Naples', 'Bari')
  and ev.date_iso > '2023-01-01'
group by ev.city, ev.venue, ev.arena_capacity, ev.arena_outside_inside, ev.arena_type
order by event_count desc;

select * from artist_categories where category = 'Category A' order by shows desc;

select * from arenas where country = 'Italy';

select * from venues where city = 'Bari;
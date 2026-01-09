-- count free models
select 
	*
from
	openrouter.models m
where 
	m.is_free = false;
-- non-free models and average cost < 5.625/10
select 
	*
from
	openrouter.models m
where
	m.is_free = false
	and avg_cost_per_1m < 0.5625;
-- count non-free models with average cost < 5.625/10 and less than 10B parameters and also available param data
select 
	*
from
	openrouter.models m
where
	m.is_free = false
	and avg_cost_per_1m < 0.5625
	and param_size < 10
	and param_size is not null;

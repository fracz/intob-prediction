<?php

$prices = array();

for($year = 2004; $year < 2014; $year++){
	echo "Fetching data for year $year..." . PHP_EOL;
	$url = 'http://www.orlen.pl/PL/DLABIZNESU/HURTOWECENYPALIW/Strony/Archiwum.aspx?Fuel=ONEkodiesel&Year=' . $year;
	$data = file_get_contents($url);
	preg_match_all('#(\d\d\-\d\d\-\d\d\d\d).*?Price">(.+?)</#isu', $data, $matches);
	for($i = 0; $i < count($matches[1]); $i++){
		$date = $matches[1][$i];
		$price = $matches[2][$i];
		$price = preg_replace('#[^\d]+#', '.', $price);
		$prices[strtotime($date)] = floatval($price);
	}
}

ksort($prices);

$contents = '';
foreach($prices as $timestamp => $price){
	$contents .= date('Y-m-d', $timestamp) . ',' . $price . PHP_EOL;
}

file_put_contents("oil.csv", $contents);
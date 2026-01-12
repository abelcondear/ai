//create model
const fs = require('node:fs');

function getRndInt(max, min) {
	return Math.ceil( Math.random() * (max - min) + min )
}

function createLine(row_id) {
	const bike = {
		prefix_len: 4,
		suffix_len: 4
	};

	const hhmm = {
		hour_len: 2,
		minute_len: 2
	};

	const label = {
		first: "09:00 AM-10:00 AM", 
		second: "10:00 AM-12:00 PM", 
		third: "12:00 PM-02:00 PM", 
		fourth: "02:00 PM-04:00 PM", 
		fifth: "04:00 PM-06:00 PM", 
		sixth: "06:00 PM-07:00 PM",
		
		morning:"morning",
		midday:"midday",
		afternoon:"afternoon",
		evening:"evening"
	};

	let prefix = ( "00" + getRndInt(110, 10) );

	prefix = prefix.substring(prefix.length - bike.prefix_len, prefix.length);

	let suffix = "";

	for ( n = 0; n < bike.suffix_len; n ++ )
	{	
		suffix += String.fromCharCode(getRndInt(90, 65));
	}

	let id = row_id;
	let bike_model = prefix.concat("-").concat(suffix); //random-number - random-string

	let quantity = getRndInt(4, 1);

	let h = 9 + getRndInt(10, 0);
	let hh = ("0" + ((h > 12) ? h - 12:h)); //09 AM to 7 PM //format number to be 9 - 12 and 1 - 7
	let mm = ("0" + ( (h != 19) ? getRndInt(59, 0):0    )); //make sure that the maximum value will be 6:59
	let rr = (Number(hh) == 12 || ( Number(hh) >= 1 && Number(hh) <= 7 )) ? " PM":" AM";

	hh = hh.substring(hh.length - hhmm.hour_len, hh.length);
	mm = mm.substring(mm.length - hhmm.minute_len, mm.length);

	let hour = hh.concat(":").concat(mm).concat(rr);

	let date = new Date().getFullYear() + "-01-01";

	let from_to = "";
	if (Number(hh) >= 9 && Number(hh) < 10) { from_to = label.first; }
	if (Number(hh) >= 10 && Number(hh) < 12) { from_to = label.second; }
	if (Number(hh) == 12 || ( Number(hh) >= 1 && Number(hh) < 2 ) ) { from_to = label.third; }
	if (Number(hh) >= 2 && Number(hh) < 4) { from_to = label.fourth; }
	if (Number(hh) >= 4 && Number(hh) < 6) { from_to = label.fifth; }
	if (Number(hh) >= 6 && Number(hh) <= 7) { from_to = label.sixth; }


	let shift = "";
	if (from_to == label.first) { shift = label.morning; }
	if (from_to == label.second) { shift = label.morning; }
	if (from_to == label.third) { shift = label.midday; }
	if (from_to == label.fourth) { shift = label.afternoon; }
	if (from_to == label.fifth) { shift = label.afternoon; }
	if (from_to == label.sixth) { shift = label.evening; }

	let arr_weather = ["hot","cold","rainy"];	
	let weather = arr_weather[getRndInt(2,0)];

	let arr_season = ["summer", "winter", "autumn", "spring"];
	let season = arr_season[getRndInt(3,0)];

	let data = [
		id,
		bike_model,
		quantity, 
		hour, 
		date,
		from_to, 
		shift, 
		weather, 
		season
	];

	return data.join(";");
}

let lines = [];
let line = "";

for (let i = 1; i < 50; i ++) {
	line = createLine(i);
	lines.push(line);
}

const content = lines.join("\r\n");

fs.writeFile('input.csv', content, err => {
  if (err) {
    console.error(err);
  } else {
	// file written successfully
  }
});

fs.readFile('input.csv', 'utf8', (err, data) => {
  if (err) {
    console.error(err);
    return;
  }
  //console.log(data);
  lines = data.split("\r\n");
  //console.log(lines);
});

console.log("\nProcess finished.");

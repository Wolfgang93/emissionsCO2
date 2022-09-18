DROP TABLE IF EXISTS`global_power_by_country`;                      
CREATE TABLE IF NOT EXISTS `global_power_by_country`(
	`iso_code` VARCHAR(30),
    `country` VARCHAR(30),
    `name` VARCHAR(240),
	`latitude` DECIMAL (10,6),
    `longitude` DECIMAL (10,6),
    `fuel` VARCHAR (30));
LOAD DATA INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\CO2\\global_power_db_filtred_by_country.csv'
INTO TABLE `global_power_by_country`
FIELDS TERMINATED BY ',' ENCLOSED BY '' ESCAPED BY '' 
LINES TERMINATED BY '\n' IGNORE 1 LINES;
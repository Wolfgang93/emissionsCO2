DROP TABLE IF EXISTS `countries`;
CREATE TABLE IF NOT EXISTS `countries`(
	`country_id` INT NOT NULL AUTO_INCREMENT,
    `country` VARCHAR (60),
    PRIMARY KEY	(country_id) );

INSERT INTO `countries` (`country`) 
SELECT DISTINCT `country`
FROM energy_co2_by_country;

SELECT * FROM countries;

DROP TABLE IF EXISTS `continents`;
CREATE TABLE IF NOT EXISTS `continents`(
	`continent_id` INT NOT NULL AUTO_INCREMENT,
    `continent` VARCHAR (60),
    PRIMARY KEY	(continent_id) );

INSERT INTO `continents` (`continent`) 
SELECT DISTINCT `country`
FROM energy_consumption_source_by_continent;

SELECT * FROM continents;
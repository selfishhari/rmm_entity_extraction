

CREATE USER 'newuser'@'localhost' IDENTIFIED BY 'password';

GRANT ALL PRIVILEGES ON * . * TO 'newuser'@'localhost';

USE rootvj_db_shop_rmm;


CREATE TABLE `cs_entity` (
  `user_id` int(100),
  `date` longtext COLLATE utf8mb4_unicode_ci,
  `tag` longtext COLLATE utf8mb4_unicode_ci,
  `brand` longtext COLLATE utf8mb4_unicode_ci,
  `model` longtext COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `service` longtext COLLATE utf8mb4_unicode_ci,
  `message_id` int(100)
)


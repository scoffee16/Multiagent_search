# orders 订单表数据字典

> 库：`deep_search`　表：`orders`　引擎：InnoDB　字符集：utf8mb4

## 1. 表结构

| 字段 | 类型 | 允许空 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| id | int | 否 | 自增 | 主键 |
| order_no | varchar(32) | 否 | - | 订单号，业务唯一键 |
| product_name | varchar(128) | 否 | - | 商品名称 |
| category | varchar(64) | 是 | NULL | 品类，见第 2 节枚举 |
| price | decimal(10,2) | 否 | - | 商品单价（元） |
| quantity | int | 否 | 1 | 购买数量 |
| total_amount | decimal(10,2) | 否 | - | 订单总金额（元）= price × quantity |
| customer_name | varchar(64) | 是 | NULL | 客户姓名（演示数据为化名） |
| status | varchar(16) | 是 | `completed` | 订单状态，默认 completed |
| created_at | datetime | 是 | CURRENT_TIMESTAMP | 下单时间 |

## 2. 枚举与编码规则

**category（品类）**：`数码电子`、`生活家电`、`美妆护肤`、`运动鞋服`

**status（订单状态）**：

| 取值 | 含义 | 是否计入 GMV |
| --- | --- | --- |
| completed | 已完成 | 是 |
| paid | 已支付未发货 | 是 |
| cancelled | 已取消 | 否 |
| refunded | 已退款 | 否 |

当前演示数据里 8 条订单全部是 `completed`。

**order_no（订单号）规则**：`ORD` + 4 位年份 + 3 位流水号，例如 `ORD2026001`。

## 3. 示例数据

| order_no | product_name | category | price | quantity | total_amount | status |
| --- | --- | --- | --- | --- | --- | --- |
| ORD2026001 | iPhone 16 Pro | 数码电子 | 8999.00 | 1 | 8999.00 | completed |
| ORD2026002 | MacBook Air M4 | 数码电子 | 10499.00 | 1 | 10499.00 | completed |
| ORD2026003 | AirPods Pro 3 | 数码电子 | 1799.00 | 2 | 3598.00 | completed |

## 4. 使用注意

- 查 GMV 时建议过滤 `status NOT IN ('cancelled','refunded')`，保证口径与 SOP 一致。
- `price` 是下单时单价，不是当前售价；做价格分析要用 `price`，不要用商品主数据表（库里也没有这张表）。
- 当前库中**只有 `orders` 这一张表**，没有商品表、库存表、流量表；库存与转化率相关指标暂时无法计算，需要等上游补表。

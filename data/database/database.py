"""
MySQL数据库操作模块
封装海洋垃圾检测系统的数据库操作
"""

import json
from datetime import datetime
from typing import List, Dict, Optional, Any

import mysql.connector
from mysql.connector import Error

from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def _format_detection_record(record: Dict) -> Dict:
    """格式化检测记录"""
    if record.get('detection_duration') is not None:
        record['detection_duration'] = float(record['detection_duration'])
    if isinstance(record.get('timestamp'), datetime):
        record['timestamp'] = record['timestamp'].isoformat()
    return record

def _format_garbage_details(details: List[Dict]) -> List[Dict]:
    """格式化垃圾详情"""
    formatted_details = []
    for detail in details:
        # 解析bbox JSON
        if detail.get('bbox_json'):
            detail['bbox'] = json.loads(detail['bbox_json'])
            del detail['bbox_json']

        # 转换confidence为float
        if detail.get('confidence') is not None:
            detail['confidence'] = float(detail['confidence'])

        formatted_details.append(detail)

    return formatted_details


class MarineDebrisMySQL:
    """
    海洋垃圾检测MySQL数据库操作类
    """

    def __init__(self,
                 host: str = "localhost",
                 port: int = 3306,
                 user: str = "root",
                 password: str = "password",
                 database: str = "garbage_detection"):
        """
        初始化MySQL数据库连接

        Args:
            host: 数据库主机
            port: 数据库端口
            user: 用户名
            password: 密码
            database: 数据库名称
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.conn = None
        self._connect()

    def _connect(self):
        """连接MySQL数据库"""
        try:
            self.conn = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset='utf8mb4',
                collation='utf8mb4_unicode_ci'
            )

            if self.conn.is_connected():
                logger.info(f"成功连接到MySQL数据库: {self.database}")
            else:
                raise Error("MySQL数据库连接失败")

        except Error as e:
            logger.error(f"MySQL数据库连接失败: {e}")
            raise

    def check_connection(self) -> bool:
        """检查数据库连接状态"""
        try:
            if self.conn is None or not self.conn.is_connected():
                self._connect()
            return True
        except Error:
            return False

    def insert_detection_record(self,
                                source_type: str,
                                source_path: str,
                                total_count: int = 0,
                                processed_file_path: str = None,
                                detection_duration: float = None,
                                status: str = 'completed') -> Optional[int]:
        """
        插入检测历史记录

        Args:
            source_type: 数据源类型 (image, video, stream)
            source_path: 源文件路径或流地址
            total_count: 垃圾总数量
            processed_file_path: 处理后的文件路径
            detection_duration: 检测耗时（秒）
            status: 检测状态 (processing, completed, failed)

        Returns:
            Optional[int]: 插入记录的ID，失败返回None
        """
        if not self.check_connection():
            return None

        try:
            cursor = self.conn.cursor()

            insert_sql = """
                         INSERT INTO detection_history
                         (source_type, source_path, total_count, processed_file_path, detection_duration, status)
                         VALUES (%s, %s, %s, %s, %s, %s) \
                         """

            cursor.execute(insert_sql, (
                source_type,
                source_path,
                total_count,
                processed_file_path,
                detection_duration,
                status
            ))

            detection_id = cursor.lastrowid
            self.conn.commit()
            cursor.close()

            logger.info(f"检测记录插入成功, ID: {detection_id}")
            return detection_id

        except Error as e:
            logger.error(f"插入检测记录失败: {e}")
            return None

    def insert_garbage_details(self,
                               detection_id: int,
                               class_name: str,
                               confidence: float,
                               bbox: Dict[str, float],
                               frame_index: int = None) -> bool:
        """
        插入垃圾检测详细信息

        Args:
            detection_id: 关联的检测记录ID
            class_name: 垃圾类别名称
            confidence: 检测置信度
            bbox: 边界框坐标字典
            frame_index: 视频帧索引（视频检测时使用）

        Returns:
            bool: 插入是否成功
        """
        if not self.check_connection():
            return False

        try:
            cursor = self.conn.cursor()

            insert_sql = """
                         INSERT INTO garbage_details
                             (detection_id, class_name, confidence, bbox_json, frame_index)
                         VALUES (%s, %s, %s, %s, %s) \
                         """

            cursor.execute(insert_sql, (
                detection_id,
                class_name,
                confidence,
                json.dumps(bbox, ensure_ascii=False),
                frame_index
            ))

            self.conn.commit()
            cursor.close()

            logger.debug(f"垃圾详情插入成功, 检测ID: {detection_id}, 类别: {class_name}")
            return True

        except Error as e:
            logger.error(f"插入垃圾详情失败: {e}")
            return False

    def insert_detection_with_details(self,
                                      source_type: str,
                                      source_path: str,
                                      garbage_list: List[Dict],
                                      processed_file_path: str = None,
                                      detection_duration: float = None,
                                      frame_index: int = None) -> Optional[int]:
        """
        插入完整的检测记录和垃圾详情（事务操作）

        Args:
            source_type: 数据源类型
            source_path: 源文件路径
            garbage_list: 垃圾检测结果列表
            processed_file_path: 处理后的文件路径
            detection_duration: 检测耗时
            frame_index: 视频帧索引

        Returns:
            Optional[int]: 检测记录ID，失败返回None
        """
        if not self.check_connection():
            return None

        try:
            cursor = self.conn.cursor()

            # 开始事务
            self.conn.start_transaction()

            # 插入检测记录
            detection_id = self.insert_detection_record(
                source_type=source_type,
                source_path=source_path,
                total_count=len(garbage_list),
                processed_file_path=processed_file_path,
                detection_duration=detection_duration
            )

            if detection_id is None:
                raise Error("插入检测记录失败")

            # 插入垃圾详情
            for garbage in garbage_list:
                success = self.insert_garbage_details(
                    detection_id=detection_id,
                    class_name=garbage.get('class_name', 'unknown'),
                    confidence=garbage.get('confidence', 0.0),
                    bbox=garbage.get('bbox', {}),
                    frame_index=frame_index
                )

                if not success:
                    raise Error("插入垃圾详情失败")

            # 提交事务
            self.conn.commit()
            cursor.close()

            logger.info(f"完整检测记录插入成功, ID: {detection_id}, 垃圾数量: {len(garbage_list)}")
            return detection_id

        except Error as e:
            self.conn.rollback()
            logger.error(f"插入完整检测记录失败: {e}")
            return None

    def get_recent_detections(self, limit: int = 50) -> List[Dict]:
        """
        获取最近的检测记录

        Args:
            limit: 返回记录数量限制

        Returns:
            List[Dict]: 检测记录列表
        """
        if not self.check_connection():
            return []

        try:
            cursor = self.conn.cursor(dictionary=True)

            query_sql = """
                        SELECT id, timestamp, source_type, source_path, total_count, processed_file_path, detection_duration, status, created_at
                        FROM detection_history
                        ORDER BY timestamp DESC
                            LIMIT %s \
                        """

            cursor.execute(query_sql, (limit,))
            records = cursor.fetchall()
            cursor.close()

            # 转换Decimal类型为float
            for record in records:
                if record.get('detection_duration') is not None:
                    record['detection_duration'] = float(record['detection_duration'])
                if isinstance(record.get('timestamp'), datetime):
                    record['timestamp'] = record['timestamp'].isoformat()

            logger.debug(f"获取最近 {len(records)} 条检测记录")
            return records

        except Error as e:
            logger.error(f"查询最近检测记录失败: {e}")
            return []

    def get_detection_details(self, detection_id: int) -> Dict[str, Any]:
        """
        获取指定检测记录的详细信息

        Args:
            detection_id: 检测记录ID

        Returns:
            Dict: 包含检测记录和垃圾详情的字典
        """
        if not self.check_connection():
            return {}

        try:
            cursor = self.conn.cursor(dictionary=True)

            # 获取检测记录
            detection_sql = "SELECT * FROM detection_history WHERE id = %s"
            cursor.execute(detection_sql, (detection_id,))
            detection_record = cursor.fetchone()

            if not detection_record:
                return {}

            # 获取垃圾详情
            details_sql = """
                          SELECT id, class_name, confidence, bbox_json, frame_index, created_at
                          FROM garbage_details
                          WHERE detection_id = %s
                          ORDER BY confidence DESC \
                          """
            cursor.execute(details_sql, (detection_id,))
            garbage_details = cursor.fetchall()

            cursor.close()

            # 处理数据格式
            result = {
                'detection_record': _format_detection_record(detection_record),
                'garbage_details': _format_garbage_details(garbage_details)
            }

            return result

        except Error as e:
            logger.error(f"查询检测详情失败: {e}")
            return {}



    def get_statistics(self,
                       start_date: str = None,
                       end_date: str = None) -> Dict[str, Any]:
        """
        获取统计信息

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            Dict: 统计信息
        """
        if not self.check_connection():
            return {}

        try:
            cursor = self.conn.cursor(dictionary=True)

            # 构建查询条件
            where_conditions = []
            params = []

            if start_date:
                where_conditions.append("DATE(timestamp) >= %s")
                params.append(start_date)
            if end_date:
                where_conditions.append("DATE(timestamp) <= %s")
                params.append(end_date)

            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)

            # 总体统计
            stats_sql = f"""
            SELECT 
                COUNT(*) as total_detections,
                SUM(total_count) as total_garbage_count,
                AVG(detection_duration) as avg_duration,
                COUNT(DISTINCT DATE(timestamp)) as active_days
            FROM detection_history
            {where_clause}
            AND status = 'completed'
            """

            cursor.execute(stats_sql, params)
            overall_stats = cursor.fetchone()

            # 类别统计
            class_sql = f"""
            SELECT 
                gd.class_name,
                COUNT(*) as count,
                AVG(gd.confidence) as avg_confidence
            FROM garbage_details gd
            JOIN detection_history dh ON gd.detection_id = dh.id
            {where_clause.replace('timestamp', 'dh.timestamp') if where_clause else ''}
            {'WHERE' not in where_clause and 'WHERE dh.status = "completed"' or 'AND dh.status = "completed"'}
            GROUP BY gd.class_name
            ORDER BY count DESC
            """

            cursor.execute(class_sql, params)
            class_stats = cursor.fetchall()

            cursor.close()

            statistics = {
                'overall': {
                    'total_detections': overall_stats['total_detections'] or 0,
                    'total_garbage_count': overall_stats['total_garbage_count'] or 0,
                    'avg_duration': float(overall_stats['avg_duration']) if overall_stats['avg_duration'] else 0,
                    'active_days': overall_stats['active_days'] or 0
                },
                'by_class': class_stats,
                'period': {
                    'start_date': start_date,
                    'end_date': end_date
                }
            }

            return statistics

        except Error as e:
            logger.error(f"查询统计信息失败: {e}")
            return {}

    def get_daily_trend(self, days: int = 30) -> List[Dict]:
        """
        获取每日趋势数据

        Args:
            days: 统计天数

        Returns:
            List[Dict]: 每日趋势数据
        """
        if not self.check_connection():
            return []

        try:
            cursor = self.conn.cursor(dictionary=True)

            trend_sql = """
                        SELECT
                            DATE (timestamp) as date, COUNT (*) as detection_count, SUM (total_count) as garbage_count, AVG (detection_duration) as avg_duration
                        FROM detection_history
                        WHERE timestamp >= DATE_SUB(CURDATE() \
                            , INTERVAL %s DAY)
                          AND status = 'completed'
                        GROUP BY DATE (timestamp)
                        ORDER BY date \
                        """

            cursor.execute(trend_sql, (days,))
            trend_data = cursor.fetchall()
            cursor.close()

            # 格式化数据
            formatted_data = []
            for item in trend_data:
                formatted_data.append({
                    'date': item['date'].isoformat() if isinstance(item['date'], datetime) else str(item['date']),
                    'detection_count': item['detection_count'],
                    'garbage_count': item['garbage_count'] or 0,
                    'avg_duration': float(item['avg_duration']) if item['avg_duration'] else 0
                })

            return formatted_data

        except Error as e:
            logger.error(f"查询趋势数据失败: {e}")
            return []

    def close(self):
        """关闭数据库连接"""
        if self.conn and self.conn.is_connected():
            self.conn.close()
            logger.info("MySQL数据库连接已关闭")

    def __enter__(self):
        """支持上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时关闭连接"""
        self.close()


# 便捷函数
def get_mysql_db(host: str = "localhost",
                 port: int = 3306,
                 user: str = "root",
                 password: str = "password",
                 database: str = "garbage_detection") -> MarineDebrisMySQL:
    """
    获取MySQL数据库实例的便捷函数
    """
    return MarineDebrisMySQL(host, port, user, password, database)


# 使用示例和测试
if __name__ == "__main__":
    # 测试数据库功能
    try:
        with get_mysql_db() as db:
            # 测试数据
            test_garbage_list = [
                {
                    'class_name': 'plastic',
                    'confidence': 0.85,
                    'bbox': {'x_min': 100, 'y_min': 100, 'x_max': 200, 'y_max': 200}
                },
                {
                    'class_name': 'foam',
                    'confidence': 0.92,
                    'bbox': {'x_min': 300, 'y_min': 300, 'x_max': 400, 'y_max': 400}
                }
            ]

            # 插入测试记录
            detection_id = db.insert_detection_with_details(
                source_type='image',
                source_path='/path/to/test/image.jpg',
                garbage_list=test_garbage_list,
                detection_duration=1.5
            )

            if detection_id:
                print(f"测试数据插入成功, ID: {detection_id}")

                # 测试查询功能
                recent_records = db.get_recent_detections(5)
                print(f"最近记录数量: {len(recent_records)}")

                stats = db.get_statistics()
                print(f"总体统计: {stats['overall']['total_garbage_count']} 个垃圾")

                trend = db.get_daily_trend(7)
                print(f"7天趋势数据: {len(trend)} 天")

    except Exception as e:
        print(f"测试失败: {e}")
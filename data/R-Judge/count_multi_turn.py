#!/usr/bin/env python3
"""
统计 R-Judge 数据集中多轮对话的数据个数
多轮对话定义：contents 长度 > 1
"""
import json
from pathlib import Path
from collections import defaultdict

def count_multi_turn_dialogues(data_dir):
    """统计多轮对话数据"""
    data_dir = Path(data_dir)
    
    # R-Judge 的五个子文件夹
    subfolders = ['Application', 'Finance', 'IoT', 'Program', 'Web']
    
    # 数据类型
    data_types = ['harmful', 'benign']
    
    # 统计结果
    results = defaultdict(lambda: defaultdict(dict))
    total_stats = {
        'total_samples': 0,
        'multi_turn_samples': 0,
        'single_turn_samples': 0,
        'max_turns': 0,
        'turn_distribution': defaultdict(int)
    }
    
    print("=" * 80)
    print("R-Judge 数据集多轮对话统计")
    print("=" * 80)
    print()
    
    # 遍历所有子文件夹和数据类型
    for subfolder in subfolders:
        for data_type in data_types:
            file_path = data_dir / subfolder / f'{data_type}.json'
            
            if not file_path.exists():
                print(f"⚠️  文件不存在: {file_path}")
                continue
            
            # 读取数据
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            total = len(data)
            multi_turn = 0
            single_turn = 0
            max_turn = 0
            turn_counts = defaultdict(int)
            
            # 统计每个样本的轮数
            for item in data:
                contents = item.get('contents', [])
                num_turns = len(contents)
                
                # 统计轮数分布
                turn_counts[num_turns] += 1
                total_stats['turn_distribution'][num_turns] += 1
                
                # 更新最大轮数
                if num_turns > max_turn:
                    max_turn = num_turns
                if num_turns > total_stats['max_turns']:
                    total_stats['max_turns'] = num_turns
                
                # 统计单轮和多轮
                if num_turns > 1:
                    multi_turn += 1
                    total_stats['multi_turn_samples'] += 1
                else:
                    single_turn += 1
                    total_stats['single_turn_samples'] += 1
                
                total_stats['total_samples'] += 1
            
            # 保存结果
            results[subfolder][data_type] = {
                'total': total,
                'multi_turn': multi_turn,
                'single_turn': single_turn,
                'multi_turn_ratio': multi_turn / total * 100 if total > 0 else 0,
                'max_turns': max_turn,
                'turn_distribution': dict(turn_counts)
            }
            
            # 打印当前文件的统计
            print(f"📁 {subfolder}/{data_type}.json")
            print(f"   总样本数: {total}")
            print(f"   多轮对话: {multi_turn} ({multi_turn/total*100:.1f}%)")
            print(f"   单轮对话: {single_turn} ({single_turn/total*100:.1f}%)")
            print(f"   最大轮数: {max_turn}")
            print()
    
    # 打印汇总统计
    print("=" * 80)
    print("📊 总体统计")
    print("=" * 80)
    print(f"总样本数: {total_stats['total_samples']}")
    print(f"多轮对话: {total_stats['multi_turn_samples']} ({total_stats['multi_turn_samples']/total_stats['total_samples']*100:.1f}%)")
    print(f"单轮对话: {total_stats['single_turn_samples']} ({total_stats['single_turn_samples']/total_stats['total_samples']*100:.1f}%)")
    print(f"最大轮数: {total_stats['max_turns']}")
    print()
    
    # 打印轮数分布
    print("=" * 80)
    print("📈 对话轮数分布")
    print("=" * 80)
    sorted_turns = sorted(total_stats['turn_distribution'].items())
    for turn, count in sorted_turns:
        percentage = count / total_stats['total_samples'] * 100
        bar_length = int(percentage / 2)  # 每 2% 一个 █
        bar = '█' * bar_length
        print(f"{turn:2d} 轮: {count:5d} ({percentage:5.1f}%) {bar}")
    print()
    
    # 按子文件夹汇总
    print("=" * 80)
    print("📂 按子文件夹统计")
    print("=" * 80)
    for subfolder in subfolders:
        if subfolder not in results:
            continue
        
        subfolder_total = 0
        subfolder_multi = 0
        
        for data_type in data_types:
            if data_type in results[subfolder]:
                stats = results[subfolder][data_type]
                subfolder_total += stats['total']
                subfolder_multi += stats['multi_turn']
        
        if subfolder_total > 0:
            print(f"{subfolder:15s}: {subfolder_multi:4d}/{subfolder_total:4d} ({subfolder_multi/subfolder_total*100:5.1f}% 多轮)")
    print()
    
    # 按数据类型汇总
    print("=" * 80)
    print("🏷️  按数据类型统计")
    print("=" * 80)
    for data_type in data_types:
        type_total = 0
        type_multi = 0
        
        for subfolder in subfolders:
            if subfolder in results and data_type in results[subfolder]:
                stats = results[subfolder][data_type]
                type_total += stats['total']
                type_multi += stats['multi_turn']
        
        if type_total > 0:
            print(f"{data_type:10s}: {type_multi:4d}/{type_total:4d} ({type_multi/type_total*100:5.1f}% 多轮)")
    print()
    
    return results, total_stats


if __name__ == '__main__':
    import sys
    
    # 可以通过命令行参数指定数据目录，默认为当前目录
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    results, total_stats = count_multi_turn_dialogues(data_dir)
    
    # 保存详细结果到 JSON 文件
    output_file = Path(data_dir) / 'multi_turn_statistics.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'total_stats': {
                'total_samples': total_stats['total_samples'],
                'multi_turn_samples': total_stats['multi_turn_samples'],
                'single_turn_samples': total_stats['single_turn_samples'],
                'max_turns': total_stats['max_turns'],
                'turn_distribution': dict(total_stats['turn_distribution'])
            }
        }, f, indent=2, ensure_ascii=False)
    
    print("=" * 80)
    print(f"✅ 详细统计结果已保存到: {output_file}")
    print("=" * 80)

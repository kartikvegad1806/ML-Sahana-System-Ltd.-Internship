from classes import *

choice = int(input("Enter Choice of Bank\n1 for HDFC Bank\n2 for Axis Bank: "))
balance = int(input("Enter initial limit: "))

if choice == 1:

    h1 = HDFC_bank(balance)
    continue_choice = 1
    while(continue_choice):
        error = 0
        trans_choice = int(input("Enter choice of transaction:\n 1. Deposit\n 2. Withdraw"))
        if trans_choice == 1:
            amount = int(input("enter money to deposit: "))
            try:
                h1.deposit(amount)
            except ValueError as e:
                print(e)
                error = 1
            if error:
                break
            else:
                continue_choice = int(input("enter 1 to continue and 0 to exit"))
        elif trans_choice == 2:
            withd_money = int(input("enter money to withdraw: "))
            try:
                h1.withdraw(withd_money)
            except ValueError as e:
                print(e)
                error = 1
            if error:
                break
            else:
                continue_choice = int(input("enter 1 to continue and 0 to exit"))

elif choice == 2:

    h1 = Axis(balance)
    continue_choice = 1
    while(continue_choice):
        error = 0
        trans_choice = int(input("Enter choice of transaction:\n 1. Deposit\n 2. Withdraw"))
        if trans_choice == 1:
            amount = int(input("enter money to deposit: "))
            try:
                h1.deposit(amount)
            except ValueError as e:
                print(e)
                error = 1
            if error:
                break
            else:
                continue_choice = int(input("enter 1 to continue and 0 to exit"))
        elif trans_choice == 2:
            withd_money = int(input("enter money to withdraw: "))
            try:
                h1.withdraw(withd_money)
            except ValueError as e:
                print(e)
                error = 1
            if error:
                break
            else:
                continue_choice = int(input("enter 1 to continue and 0 to exit"))
